## Project context (MRAIA)

- The `mraia` MCP server holds this project's knowledge base. Call `project_brief` at the start of a session, and use
  `search_knowledge` / `read_topic` before asking us for project context.
- M1 to M5 in the proposal are recommendations, not requirements. If you see a better modification, say so with
  reasons and record it with `suggest_modification`.
- Log every training or evaluation run with `log_experiment` and `update_experiment`. Use `record_decision` only for
  decisions we actually made, and `record_finding` for things learned (with evidence).
- Never change the method, data, loss or hyperparameters silently to get past an error. Stop and explain.
- Run scripts from the repo root. Start long training runs in a separate terminal, after a short smoke test and report the user if any long training or data is required.
