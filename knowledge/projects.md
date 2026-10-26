# Projects

## Project: ticketherd
Open-source Rust CLI, github.com/yaronlavi1992/ticketherd (started October 2026).
- Gives every ticket its own git worktree, port and coding agent: `ticketherd start 42 --implement`.
- A loop watches pull requests and acts on what they need, using six idempotent trigger
  rules capped per poll, while every irreversible step still stops and asks a human.
- `ticketherd verify` serves the branch and drives it in a real browser to collect evidence
  (screenshots, console errors, accessibility, pixel diffs).
- Composes existing tools instead of forking them: worktrunk for worktrees and ports,
  agent-browser for browser automation, herdr for terminal panes, and the GitHub CLI.

## Project: Ask my CV (this repository)
Python agent that answers recruiters' questions about Yaron's experience.
- Answers only from the facts in this repository and cites the section each fact came from.
- Declines questions the facts don't cover instead of guessing, and ignores instructions
  hidden inside questions.
- Works with any OpenAI-compatible model provider, including a local model.
- Ships with an evaluation suite that checks accuracy, refusals and prompt-injection
  resistance; results are published in this README.

## Project: Architect Haven
3D visualization tool built with React and Three.js (side project).
