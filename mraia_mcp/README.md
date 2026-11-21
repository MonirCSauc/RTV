# MRAIA context server for Claude Code

A small MCP server that gives Claude Code the project's context: what MRAIA is, how the RTV pipeline works, the
repo map and its hazards, the proposal's recommended modifications, metrics, compute and recording plans. It also
keeps shared logs of experiments, decisions, findings and suggested modifications.

The knowledge is plain markdown in `knowledge/`. Edit those files directly; the server re-reads them on every call.

## Setup (Windows, once per PC)

1. **Copy the files into the repo root** (`D:\RTV`): the `mraia_mcp\` folder and `.mcp.json`.
2. **Install the MCP library** into the Python that is on your PATH (it needs no PyTorch; any Python 3.10+):
   ```
   python -m pip install "mcp>=1.10"
   ```
3. **Check it works** from the repo root:
   ```
   python mraia_mcp\server.py --check
   ```
   It should list 16 topics and end with `OK`.
4. **Open Claude Code in `D:\RTV`.** It reads `.mcp.json` and asks once whether to allow the `mraia` server: allow it.
   Run `/mcp` to confirm `mraia` shows as connected.
5. **Add the lines in `CLAUDE_md_snippet.md` to the repo's `CLAUDE.md`** so Claude uses the server every session.
6. Start a session with the prompt `/mcp__mraia__session_start`.

If `python` is not found or points to the wrong install, replace `"python"` in `.mcp.json` with the full path, for
example `"C:\\Users\\<you>\\miniconda3\\python.exe"`, or register it by hand:

```
claude mcp add mraia --scope project -- python D:/RTV/mraia_mcp/server.py
```

## What Claude gets

| Tool | Use |
|---|---|
| `project_brief` | Overview, working rules, topic list, recent experiments. Called at session start. |
| `list_topics`, `read_topic` | Read a topic, or one section of it (`section="..."`). |
| `search_knowledge` | Keyword search across all topics and the experiment log, with line numbers. |
| `get_proposal` | One of M1 to M5, plus any logged suggestions about it. |
| `define` | Glossary lookup (VM, SDP, SSDP, BEV, FiLM ...). |
| `log_experiment`, `update_experiment`, `list_experiments` | Shared experiment log in `knowledge/experiments.jsonl`, with the git commit recorded automatically. |
| `record_decision`, `record_finding` | Append dated entries to `decisions.md` and `findings.md`. |
| `suggest_modification` | Record a better or additional modification for the team to review (`suggestions.md`). |

Prompts (type them as slash commands in Claude Code):

- `/mcp__mraia__session_start`: load the context and summarise where the project stands.
- `/mcp__mraia__plan_experiment`: plan one experiment (including at least one alternative to the proposal) before running it.
- `/mcp__mraia__review_proposal`: critically review one of M1 to M5 against the code and look for better options.

## The proposal is not a cage

M1 to M5 are recommendations from the proposal stage. The server tells Claude to judge ideas against the research
question and constraints, and to propose and log better modifications when it sees them. Accepted suggestions are
moved into `proposals.md` by the team, with a line in `decisions.md`.

## Keeping it useful

- Commit `mraia_mcp/` (including `knowledge/experiments.jsonl`) to git so the whole team shares the same context and
  logs. If two branches both log experiments, keep both sets of lines when merging (one line per run).
- Add a new topic by creating a markdown file in `knowledge/` with this header:
  ```
  ---
  title: Short title
  summary: One line saying what is inside and when to read it.
  ---
  ```
- Keep topics factual and short. Update `overview.md` when the project status changes.
- `fork_changes.md` was written on 1 Nov 2026; refresh it from `git diff upstream/main` when the fork moves on.
- Environment variables (optional): `MRAIA_KNOWLEDGE_DIR` to point at another knowledge folder. Claude Code sets
  `CLAUDE_PROJECT_DIR`, which the server uses to find the repo for git commits.
