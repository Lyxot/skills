---
name: call-agents
description: Start, resume and read other coding agents — Claude, Codex, Cursor and Antigravity (agy) — for reviews, second opinions, research or delegated work. Use whenever a task says to launch a reviewer or another agent, to pick which model to give a job, or to read back an agent's result or session ID. Covers the model allowlist and effort per job difficulty, each CLI's exact start and resume commands, where each one leaves its final message and session ID, and the rules for briefing and waiting on agents.
---

# call-agents

## Models

Only these models, at these efforts. Pick by how hard the job is, not by habit.

| Family | Super easy jobs | Daily use | Very hard jobs |
|---|---|---|---|
| Claude | `haiku-5.5` xhigh | `opus-5.5` medium / high | `fable-5.2` high |
| Codex | `gpt-6-luna` xhigh | `gpt-6.1-sol` medium / high | `gpt-6-astra` high |
| Cursor | — | `auto` | — |
| Antigravity | — | `gemini-4-hydrogen` high | `gemini-4-argon` / `gemini-4-carbon` high |

- Super easy: lookups, renames, summarising a log, mechanical edits. Very hard: subtle concurrency or numerics, design under real constraints, a bug others missed. Everything else is daily use; take high over medium when the job is long or a mistake is costly.
- agy also lists Claude, GPT-OSS and Flash models; never use those through agy. `gemini-4-carbon` appears only on some accounts: use it when `agy models` lists it, otherwise `gemini-4-argon`.
- A task's own instructions (a review workflow, say) may narrow this list; they never widen it.

## Built-in tools first

When the orchestrator is Claude or Codex, start its own family's models with its built-in agent tools, not the CLI: they keep the result in the conversation and cost no process management.

- **Claude** (Claude Code): the `Agent` tool with `model` `haiku`, `opus` or `fable` and `effort` from the table; resume with `SendMessage` to the agent's ID.
- **Codex**: `spawn_agent` for a new agent, `followup_task` to resume one, with the model and reasoning effort from the table. Model and effort overrides need `fork_turns="none"`; the default (`"all"`) rejects them.

Every other family, and any orchestrator without such tools, uses the CLIs below.

## CLI commands

Set `model` and `effort` from the table before each call; the commands read them and refuse to run while either is unset, so a copied command never falls back to some model. Run each from the workspace root (where `AGENTS.md` lives), never from a reviewer worktree, so the agent loads the same rules. Give each agent, and each resumed turn, its own log and output file; keep the logs, since the session ID is recovered from them. `</dev/null` everywhere: the prompt is an argument, and an open stdin makes some CLIs wait.

### Codex

```bash
# set first, from the table: model=gpt-6-luna|gpt-6.1-sol|gpt-6-astra effort=medium|high|xhigh
codex -C "$root" exec --model "${model:?pick from the table}" -c "model_reasoning_effort=\"${effort:?pick from the table}\"" \
    --yolo --skip-git-repo-check --json --output-last-message "$out" "$prompt" </dev/null > "$log"
session="$(jq -rn -R '[inputs | fromjson? | select(.type == "thread.started") | .thread_id] | first // empty' "$log")"

codex -C "$root" exec resume --model "${model:?pick from the table}" -c "model_reasoning_effort=\"${effort:?pick from the table}\"" \
    --yolo --skip-git-repo-check --json --output-last-message "$out2" "$session" "$prompt2" </dev/null > "$log2"
```

- Result: the `--output-last-message` file. Session: `thread_id` of the first `thread.started` event in the log (the log's first line is not JSON; the expression skips it).
- `--skip-git-repo-check` because the workspace root need not be a repository. Never `--ephemeral` (it cannot be resumed) or `--last`.

### Cursor

```bash
# set first: model=auto (Cursor's only allowed model; it takes no effort)
cursor-agent -p --output-format json --model "${model:?pick from the table}" --force --trust --workspace "$root" "$prompt" </dev/null > "$log"
session="$(jq -r '.session_id // empty' "$log")"; jq -r '.result' "$log" > "$out"

cursor-agent -p --output-format json --model "${model:?pick from the table}" --force --trust --workspace "$root" \
    --resume "$session" "$prompt2" </dev/null > "$log2"
jq -r '.result' "$log2" > "$out2"
```

- Result `.result`, session `.session_id`, in one JSON object. Without `--force` shell commands are blocked, so the agent cannot run `git diff` or tests. Never `--continue`.

### Claude

```bash
# set first, from the table: model=claude-haiku-5-5|claude-opus-5-5|claude-fable-5-2 effort=medium|high|xhigh
claude -p --model "${model:?pick from the table}" --effort "${effort:?pick from the table}" --output-format json --dangerously-skip-permissions \
    "$prompt" </dev/null > "$log"
session="$(jq -r '.session_id // empty' "$log")"; jq -r '.result' "$log" > "$out"

claude -p --model "${model:?pick from the table}" --effort "${effort:?pick from the table}" --output-format json --dangerously-skip-permissions \
    --resume "$session" "$prompt2" </dev/null > "$log2"
jq -r '.result' "$log2" > "$out2"
```

- Result `.result`, session `.session_id`. Check `.is_error` first: a failed run still prints a result, such as an expired login, and still has a session ID.
- An expired login (`OAuth session expired`) needs the user: they run `claude` once and `/login`. Agents cannot fix it.

### Antigravity (agy)

```bash
# set first, from the table: model=gemini-4-hydrogen|gemini-4-argon|gemini-4-carbon effort=high
agy --model "${model:?pick from the table}-${effort:?pick from the table}" --dangerously-skip-permissions --output-format json \
    --print-timeout 0 -p "$prompt" </dev/null > "$log"
session="$(jq -r '.conversation_id // empty' "$log")"; jq -r '.response' "$log" > "$out"

agy --conversation "$session" --model "${model:?pick from the table}-${effort:?pick from the table}" --dangerously-skip-permissions \
    --output-format json --print-timeout 0 -p "$prompt2" </dev/null > "$log2"
jq -r '.response' "$log2" > "$out2"
```

- `-p` takes the prompt as its value, so it goes last, directly before the prompt; anything after it is taken for the prompt and rejected.
- The effort is part of the model name, so the command joins them (`gemini-4-hydrogen-high`); `agy models` lists the names.
- Result `.response`, session `.conversation_id`; `.status` is `SUCCESS` on success. Without `--dangerously-skip-permissions`, every shell command is denied in print mode and listed under `.denied_actions`, and the response comes back empty.
- agy truncates a long `AGENTS.md` it loads. When the task depends on rules past the start of the file, tell it to read `AGENTS.md` in full.

## Rules

- **Brief the agent, don't restate the workspace.** It loads `AGENTS.md` itself (agy only partly, above). Spend the prompt on the goal, the files, the decisions so far, and what done looks like.
- **Name the session.** Keep every agent's session ID next to its log and output, and resume by that ID only. Never resume "the last" session (`--last`, `--continue`, `-c`): parallel agents make that someone else's.
- **Approvals are off, so limits are behavioral.** `--yolo`, `--force` and `--dangerously-skip-permissions` let the agent run anything. Accept that only in this trusted workspace, never on an untrusted checkout. A read-only agent, such as a reviewer, is told so in its prompt, and told not to start agents of its own.
- **Wait without blocking the user.** From the main conversation, start a CLI agent with the harness's background mode, which notifies you when the command exits (Claude Code: `Bash` with `run_in_background`); a shell `&` gives no such notice. Never sleep in the foreground. A subagent cannot yield, so inside one, wait with a single blocking call.
- **Read the output file, check the status.** An exit status of 0 can still carry an error result (Claude `.is_error`, agy `.status` or `.denied_actions`, an empty Codex output). A failed or empty run says nothing about the work: rerun it, or report that it failed.
- **Treat results as claims.** Verify an agent's findings against the code before acting on them, and say which agent and model produced what you report.
- **Don't interrupt.** Leave a running agent alone unless it is stuck; feedback goes in a resumed turn, not mid-run.
