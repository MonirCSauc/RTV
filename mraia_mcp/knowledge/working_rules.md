---
title: Working rules for Claude on this repo
summary: How to work on this research codebase: think beyond the proposal, no silent scope changes, protect evaluation code, smoke tests before long runs, cite code, log experiments.
---

# Working rules

1. **The proposal is a starting point, not a cage.** M1 to M5 are recommendations. Judge every idea against the
   research question and our constraints (16 GB GPU now, real-time inference, May 2027 deadline). If you see a better
   modification, propose it with reasons and evidence and log it with `suggest_modification`.
2. **No silent scope changes.** Never swap the method, model, dataset, loss or hyperparameters to get past an error.
   Stop and explain the problem instead.
3. **Protect evaluation.** Do not edit metric or evaluation code while comparing methods, and never tune on the
   held-out chunk.
4. **Smoke test before long runs.** Run one or two iterations on a tiny subset before any run longer than a few
   minutes. Start long training runs in the user's own terminal (tmux or a separate window), not as a background
   task inside Claude Code.
5. **Report honestly.** State what was actually run and measured. Mark estimates as estimates. If a run failed or was
   cut short, say so.
6. **Cite the code.** Back claims about the code with `file:line`. Check `hazards` before changing colour order,
   value ranges, crops or affine matrices.
7. **Run from the repo root.** Scripts use relative paths.
8. **Log what matters.** Use `log_experiment` / `update_experiment` for every training or evaluation run,
   `record_decision` for team decisions, and `record_finding` for things learned. Search the knowledge base before
   asking the user for context.
9. **Upstream vs fork.** Know which one a fact is about (`fork_changes`).
