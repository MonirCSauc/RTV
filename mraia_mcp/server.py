"""MRAIA project context server for Claude Code (Model Context Protocol, stdio).

Serves the team's knowledge base (markdown files in ./knowledge) and keeps three logs that
Claude and the team share: experiments, decisions/findings, and suggested modifications.

Run by Claude Code (see README.md). Self-check: python server.py --check
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

HERE = Path(__file__).resolve().parent
KNOWLEDGE = Path(os.environ.get("MRAIA_KNOWLEDGE_DIR") or HERE / "knowledge").resolve()
REPO_ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or HERE.parent).resolve()
EXPERIMENTS = KNOWLEDGE / "experiments.jsonl"
MAX_TEXT = 4000  # longest text accepted in one logged field
_write_lock = threading.Lock()

INSTRUCTIONS = """\
Project context for MRAIA, a real-time virtual try-on smart mirror built on the RTV research repo
(Wu, Shen, Igarashi 2025). Call project_brief once at the start of a session. Use search_knowledge or
read_topic before asking the user for project context. The proposal's modifications (M1 to M5) are
recommendations, not requirements: judge ideas against the research question and constraints, and when
you see a better modification, propose it with reasons and record it with suggest_modification.
Log every training or evaluation run with log_experiment / update_experiment, and record team decisions
and findings with record_decision / record_finding. Never invent results."""

mcp = _Server("mraia", instructions=INSTRUCTIONS)


# ---------------------------------------------------------------- knowledge files

def _today() -> str:
    return dt.date.today().isoformat()


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Return (metadata, body) for a markdown file with an optional --- front matter block."""
    meta: dict[str, str] = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip().lower()] = value.strip()
            text = text[end + 4:].lstrip("\n")
    return meta, text


def _topics() -> dict[str, Path]:
    """All markdown files in the knowledge folder, keyed by name without .md (subfolders use '/')."""
    found = {}
    for path in sorted(KNOWLEDGE.rglob("*.md")):
        name = path.relative_to(KNOWLEDGE).with_suffix("").as_posix()
        found[name] = path
    return found


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_topic(topic: str) -> tuple[str, Path] | None:
    topics = _topics()
    key = topic.strip().removesuffix(".md").replace("\\", "/").lower()
    for name, path in topics.items():
        if name.lower() == key:
            return name, path
    matches = [(n, p) for n, p in topics.items() if key and key in n.lower()]
    return matches[0] if len(matches) == 1 else None


def _topic_index() -> str:
    lines = []
    for name, path in _topics().items():
        meta, _ = _split_front_matter(_read(path))
        lines.append(f"- `{name}`: {meta.get('summary') or meta.get('title') or ''}")
    return "\n".join(lines)


def _section(body: str, heading: str) -> str | None:
    """Return the markdown section whose heading contains `heading` (case-insensitive)."""
    lines = body.splitlines()
    want = heading.strip().lower()
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m and want in m.group(2).lower():
            level = len(m.group(1))
            out = [line]
            for nxt in lines[i + 1:]:
                n = re.match(r"^(#{1,6})\s+", nxt)
                if n and len(n.group(1)) <= level:
                    break
                out.append(nxt)
            return "\n".join(out).strip()
    return None


def _append_log(path: Path, entry: str) -> None:
    with _write_lock:
        text = _read(path) if path.exists() else ""
        if not text.endswith("\n"):
            text += "\n"
        path.write_text(text + "\n" + entry.strip() + "\n", encoding="utf-8")


def _clean(value: str | None, limit: int = MAX_TEXT) -> str:
    value = (value or "").strip()
    return value if len(value) <= limit else value[:limit] + " [truncated]"


def _one_line(value: str | None, limit: int = 200) -> str:
    return " ".join(_clean(value, limit).split())


# ---------------------------------------------------------------- git and experiments

def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _git_state() -> str:
    commit = _git("rev-parse", "--short", "HEAD")
    if not commit:
        return "unknown"
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return commit + (" (uncommitted changes)" if dirty else "")


def _load_experiments() -> list[dict]:
    if not EXPERIMENTS.exists():
        return []
    runs = []
    for line in _read(EXPERIMENTS).splitlines():
        line = line.strip()
        if line:
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs


def _save_experiments(runs: list[dict]) -> None:
    EXPERIMENTS.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in runs), encoding="utf-8"
    )


def _format_metrics(metrics: dict | None) -> str:
    if not metrics:
        return ""
    return ", ".join(f"{k}={v}" for k, v in metrics.items())


def _experiment_table(runs: list[dict]) -> str:
    if not runs:
        return "No experiments logged yet."
    rows = ["| ID | Date | Status | Title | Method | Metrics | Commit |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        rows.append(
            f"| {r['id']} | {r.get('date', '')} | {r.get('status', '')} | {r.get('title', '')} | "
            f"{r.get('method', '')} | {_format_metrics(r.get('metrics'))} | {r.get('commit', '')} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------- tools: reading

@mcp.tool(structured_output=False)
def project_brief() -> str:
    """Start here. Returns the project overview, working rules, the list of knowledge topics and recent
    experiments. Call once at the start of a session."""
    parts = []
    for name in ("overview", "working_rules"):
        found = _resolve_topic(name)
        if found:
            parts.append(_split_front_matter(_read(found[1]))[1].strip())
    parts.append("# Knowledge topics (use read_topic)\n\n" + _topic_index())
    recent = list(reversed(_load_experiments()))[:5]
    parts.append("# Recent experiments\n\n" + _experiment_table(recent))
    parts.append(f"Repo root: {REPO_ROOT}  ·  commit: {_git_state()}  ·  today: {_today()}")
    return "\n\n---\n\n".join(parts)


@mcp.tool(structured_output=False)
def list_topics() -> str:
    """List every knowledge topic with a one-line summary."""
    return _topic_index() or "The knowledge folder is empty."


@mcp.tool(structured_output=False)
def read_topic(topic: str, section: str = "") -> str:
    """Read one knowledge topic, e.g. 'pipeline', 'proposals', 'hazards', 'repo_map', 'training',
    'metrics'. Optionally pass `section` (part of a heading) to return only that section."""
    found = _resolve_topic(topic)
    if not found:
        return f"No topic named '{topic}'. Available topics:\n" + _topic_index()
    name, path = found
    meta, body = _split_front_matter(_read(path))
    if section:
        part = _section(body, section)
        if part is None:
            headings = [h for h in body.splitlines() if h.startswith("#")]
            return f"No section matching '{section}' in `{name}`. Headings:\n" + "\n".join(headings)
        body = part
    return f"[topic: {name}]\n\n{body}"


@mcp.tool(structured_output=False)
def search_knowledge(query: str, max_results: int = 8) -> str:
    """Keyword search across all knowledge topics and the experiment log. Returns the best matching
    passages with topic, heading and line number. Use for questions like 'where is compositing done',
    'VGG offset bug', 'ConvLSTM state', 'BEDLAM'."""
    words = [t for t in re.findall(r"[\w./-]+", query.lower()) if len(t) > 1]
    terms = [t for t in words if t not in _STOPWORDS] or words
    if not terms:
        return "Give at least one search word."
    phrase = " ".join(query.lower().split())
    hits = []
    sources = [(name, _read(path)) for name, path in _topics().items()]
    if EXPERIMENTS.exists():
        sources.append(("experiments", _read(EXPERIMENTS)))
    for name, text in sources:
        body = _split_front_matter(text)[1]
        if name == "experiments":
            units = [("experiment log", i + 1, line) for i, line in enumerate(text.splitlines()) if line.strip()]
        else:
            units = list(_units(text))
        for heading, line_no, unit in units:
            low, head = unit.lower(), heading.lower()
            present = [t for t in terms if t in low or t in head]
            if not present:
                continue
            score = 4 * len(present) + min(sum(low.count(t) for t in present), 6)
            score += 2 * sum(1 for t in present if t in head)
            if len(present) == len(terms):
                score += 5
            if len(terms) > 1 and phrase in " ".join(low.split()):
                score += 8
            score -= len(unit) / 800  # prefer short, specific passages
            if unit.startswith("#"):  # a heading hit: show the start of its section
                unit = _section(body, heading) or unit
            hits.append((score, name, heading, line_no, unit))
    if not hits:
        return f"No matches for '{query}'. Try other words, or list_topics."
    hits.sort(key=lambda h: -h[0])
    out = []
    for _, name, heading, line, unit in hits[: max(1, min(max_results, 25))]:
        snippet = unit if len(unit) <= 700 else unit[:700] + " ..."
        out.append(f"### {name} · {heading or '(top)'} · line {line}\n{snippet}")
    return "\n\n".join(out)


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "of", "in", "on", "to", "for", "and", "or", "how", "what", "where",
    "which", "why", "when", "does", "do", "done", "it", "we", "our", "with", "be", "can", "should", "this", "that",
}


def _units(text: str):
    """Split a markdown file into searchable units: headings, table rows, list items (with their
    indented continuation lines) and paragraphs. Yields (heading, line number, text)."""
    _, body = _split_front_matter(text)
    offset = text.count("\n", 0, len(text) - len(body))
    heading, unit, start, in_code = "", [], 0, False
    for i, line in enumerate(body.splitlines()):
        line_no = offset + i + 1
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
        new_unit = (
            not in_code and (
                line.startswith("#")
                or stripped.startswith("|")
                or re.match(r"^(\s{0,3})([-*]|\d+\.)\s", line) is not None
                or not stripped
            )
        )
        if new_unit or not unit:
            if unit:
                yield heading, start, "\n".join(unit)
            unit, start = [], line_no
            if line.startswith("#") and not in_code:
                heading = line.lstrip("#").strip()
                yield heading, line_no, line
                continue
            if not stripped or re.match(r"^\|[\s:|-]+\|$", stripped):
                continue
        unit.append(line)
    if unit:
        yield heading, start, "\n".join(unit)


@mcp.tool(structured_output=False)
def get_proposal(modification: str) -> str:
    """Return one proposed modification (M1 to M5) with idea, code it touches, risks and tests, plus any
    logged suggestions that mention it. Remember these are recommendations, open to better alternatives."""
    m = re.search(r"(\d)", modification)
    if not m or not 1 <= int(m.group(1)) <= 5:
        return "Use M1, M2, M3, M4 or M5."
    key = f"M{m.group(1)}"
    found = _resolve_topic("proposals")
    if not found:
        return "The proposals topic is missing."
    body = _split_front_matter(_read(found[1]))[1]
    part = _section(body, key + ":") or _section(body, key) or f"No section for {key}."
    note = ("Status: recommendation from the proposal stage, not a commitment. "
            "If you see a better route to the same goal, say so and use suggest_modification.")
    related = []
    sugg = _resolve_topic("suggestions")
    if sugg:
        log = _split_front_matter(_read(sugg[1]))[1]
        for block in re.split(r"\n(?=### )", log):
            if block.startswith("### ") and re.search(rf"\b{key}\b", block, re.IGNORECASE):
                related.append(block.strip())
    extra = "\n\n## Logged suggestions mentioning " + key + "\n\n" + "\n\n".join(related) if related else ""
    return f"{note}\n\n{part}{extra}"


@mcp.tool(structured_output=False)
def define(term: str) -> str:
    """Define a project term or abbreviation (VM, SDP, SSDP, IUV, BEV, SMPL, BodyMap, ReGarSyn, FiLM ...)."""
    found = _resolve_topic("glossary")
    want = term.strip().lower()
    if found and want:
        entries = [l for l in _read(found[1]).splitlines() if l.startswith("- **")]
        exact = [e for e in entries if want in re.findall(r"\*\*(.+?)\*\*", e)[0].lower()]
        if exact:
            return "\n".join(exact)
        loose = [e for e in entries if want in e.lower()]
        if loose:
            return "\n".join(loose[:5])
    return search_knowledge(term, max_results=3)


# ---------------------------------------------------------------- tools: experiments

@mcp.tool(structured_output=False)
def log_experiment(
    title: str,
    goal: str,
    method: str = "",
    command: str = "",
    config: str = "",
    dataset: str = "",
    status: str = "planned",
    metrics: dict[str, float] | None = None,
    notes: str = "",
) -> str:
    """Log a training or evaluation run (planned, running, done, failed or abandoned). Records the
    git commit automatically. `method` can be M1 to M5, 'baseline', or a new idea. Returns the new ID."""
    with _write_lock:
        runs = _load_experiments()
        numbers = [int(r["id"].split("-")[1]) for r in runs if re.match(r"EXP-\d+$", r.get("id", ""))]
        run = {
            "id": f"EXP-{(max(numbers) + 1) if numbers else 1:03d}",
            "date": _today(),
            "title": _one_line(title),
            "goal": _clean(goal),
            "method": _one_line(method, 80),
            "command": _clean(command),
            "config": _clean(config),
            "dataset": _clean(dataset, 500),
            "status": _one_line(status, 30) or "planned",
            "metrics": metrics or {},
            "notes": [f"{_today()}: {_clean(notes)}"] if notes.strip() else [],
            "commit": _git_state(),
        }
        runs.append(run)
        _save_experiments(runs)
    return f"Logged {run['id']} ({run['status']}) at commit {run['commit']}."


@mcp.tool(structured_output=False)
def update_experiment(
    experiment_id: str,
    status: str = "",
    metrics: dict[str, float] | None = None,
    notes: str = "",
) -> str:
    """Update a logged experiment: new status (running, done, failed, abandoned), measured metrics
    (merged into existing ones) and/or a dated note. Report only numbers that were actually measured."""
    with _write_lock:
        runs = _load_experiments()
        for run in runs:
            if run.get("id", "").lower() == experiment_id.strip().lower():
                if status.strip():
                    run["status"] = _one_line(status, 30)
                if metrics:
                    run.setdefault("metrics", {}).update(metrics)
                if notes.strip():
                    run.setdefault("notes", []).append(f"{_today()}: {_clean(notes)}")
                run["updated"] = _today()
                _save_experiments(runs)
                return f"Updated {run['id']}: status={run['status']}; metrics: {_format_metrics(run.get('metrics')) or 'none'}."
    return f"No experiment with ID '{experiment_id}'. Use list_experiments."


@mcp.tool(structured_output=False)
def list_experiments(status: str = "", method: str = "", limit: int = 20, details: bool = False) -> str:
    """List logged experiments, newest first. Filter by status or method (e.g. 'M5', 'baseline').
    Set details=true to include goal, command, config and notes."""
    runs = list(reversed(_load_experiments()))
    if status:
        runs = [r for r in runs if r.get("status", "").lower() == status.lower()]
    if method:
        runs = [r for r in runs if method.lower() in r.get("method", "").lower()]
    runs = runs[: max(1, min(limit, 200))]
    if not details:
        return _experiment_table(runs)
    return "\n\n".join(json.dumps(r, ensure_ascii=False, indent=2) for r in runs) or "No experiments match."


# ---------------------------------------------------------------- tools: decisions, findings, suggestions

def _log_path(topic: str) -> Path:
    found = _resolve_topic(topic)
    return found[1] if found else KNOWLEDGE / f"{topic}.md"


@mcp.tool(structured_output=False)
def record_decision(decision: str, reason: str, alternatives: str = "") -> str:
    """Record a decision the team made (not a proposal), with the reason and alternatives considered.
    Only call this when the user has actually decided."""
    entry = f"### {_today()} · {_one_line(decision)}\n- Reason: {_clean(reason)}"
    if alternatives.strip():
        entry += f"\n- Alternatives considered: {_clean(alternatives)}"
    _append_log(_log_path("decisions"), entry)
    return "Decision recorded in decisions.md."


@mcp.tool(structured_output=False)
def record_finding(finding: str, evidence: str = "", tags: str = "") -> str:
    """Record something learned about the code, data, papers or experiments, with its evidence
    (file:line, experiment ID, paper section)."""
    entry = f"### {_today()} · {_one_line(finding)}"
    if evidence.strip():
        entry += f"\n- Evidence: {_clean(evidence)}"
    if tags.strip():
        entry += f"\n- Tags: {_one_line(tags)}"
    _append_log(_log_path("findings"), entry)
    return "Finding recorded in findings.md."


@mcp.tool(structured_output=False)
def suggest_modification(
    title: str,
    relation: str,
    rationale: str,
    evidence: str = "",
    effort: str = "",
    risks: str = "",
) -> str:
    """Propose a modification that could serve the research question better than, or in addition to,
    the proposal's M1 to M5. `relation` says what it replaces or extends (e.g. 'replaces M2',
    'alternative to M5', 'new'). The team reviews suggestions; this does not change the plan."""
    entry = (
        f"### {_today()} · {_one_line(title)}\n"
        f"- Relation to the proposal: {_one_line(relation)}\n"
        f"- Why it could be better: {_clean(rationale)}"
    )
    if evidence.strip():
        entry += f"\n- Evidence: {_clean(evidence)}"
    if effort.strip():
        entry += f"\n- Effort and hardware: {_clean(effort, 500)}"
    if risks.strip():
        entry += f"\n- Risks: {_clean(risks, 1000)}"
    entry += "\n- Status: awaiting team review"
    _append_log(_log_path("suggestions"), entry)
    return "Suggestion recorded in suggestions.md for team review."


# ---------------------------------------------------------------- resources and prompts

@mcp.resource("mraia://brief", name="brief", description="Project overview, rules, topics, recent experiments")
def brief_resource() -> str:
    return project_brief()


@mcp.resource("mraia://topic/{name}", name="topic", description="One knowledge topic by name")
def topic_resource(name: str) -> str:
    return read_topic(name)


@mcp.prompt(description="Load the project context and get ready to work")
def session_start() -> str:
    return (
        "Call the mraia project_brief tool and read it. Then summarise in 5 lines where the project stands, "
        "what the open next steps are, and anything in the experiment log that needs attention. "
        "Do not change any files yet."
    )


@mcp.prompt(description="Plan one experiment before running it")
def plan_experiment(goal: str) -> str:
    return (
        f"Goal: {goal}\n\n"
        "1. Use search_knowledge and read_topic (pipeline, training, metrics, hazards, compute) and list_experiments "
        "to gather what we already know.\n"
        "2. Consider the proposal's modifications, but also at least one alternative that could reach the goal "
        "better or more cheaply. Say which you recommend and why.\n"
        "3. Write a short spec: hypothesis, exact change (files), data, metrics, a smoke test, expected time and "
        "memory on the 16 GB RTX 4080, and what result would count as success or failure.\n"
        "4. After I approve, log it with log_experiment (status 'planned')."
    )


@mcp.prompt(description="Critically review one proposed modification and look for better options")
def review_proposal(modification: str) -> str:
    return (
        f"Review {modification} from the proposal.\n"
        "1. get_proposal for its definition, then check the code it touches (cite file:line).\n"
        "2. Assess feasibility on our hardware, expected gain, risks, and how we would measure it.\n"
        "3. Look for a better or simpler alternative (code, literature, experiments). Do not assume the proposal "
        "is right. If you find one, record it with suggest_modification.\n"
        "4. Give a verdict: keep, change, or replace, with reasons."
    )


# ---------------------------------------------------------------- entry point

def _self_check() -> int:
    topics = _topics()
    print(f"Knowledge folder: {KNOWLEDGE}")
    print(f"Repo root: {REPO_ROOT} (commit: {_git_state()})")
    print(f"{len(topics)} topics: {', '.join(topics)}")
    print(f"{len(_load_experiments())} experiments logged")
    print("OK" if topics else "WARNING: no knowledge files found")
    return 0 if topics else 1


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(_self_check())
    KNOWLEDGE.mkdir(parents=True, exist_ok=True)
    mcp.run()
