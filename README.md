# AI Collections Agent Simulator

A local, rule-based simulator of a financial collections/support agent.
It classifies user utterances into intents, stores the interaction
history in SQLite, and exposes a simple CLI loop for experimenting with
the agent.

> **Day 1 scope** — intent classification, memory management, and a
> basic CLI. No ML, no external APIs, no UI framework.

## Project layout

```
ai_agent_simulator/
    __init__.py
    app.py                       # CLI entry point
    agent/
        __init__.py
        intent_classifier.py     # Rule-based intent classifier
        memory_manager.py        # SQLite-backed interaction memory
```

## Requirements

* Python 3.10+ (uses PEP 604 `|` unions and `from __future__ import annotations`)
* SQLite (bundled with Python via the `sqlite3` module)

No third-party packages are required for the Day 1 foundation. See
`requirements.txt`.

## Running the CLI

From the project root:

```bash
python -m ai_agent_simulator.app
```

Useful flags:

```bash
python -m ai_agent_simulator.app --user-id alice --db ./memory.db
```

### Example session

```
AI Collections Agent Simulator (user: 'default_user'). Type /help for commands, /quit to exit.
User: I will pay later
Detected Intent: delaying (confidence: 0.83)
User: this is not working
Detected Intent: frustrated (confidence: 0.67)
User: okay thanks
Detected Intent: cooperative (confidence: 0.67)
User: /history
History for 'default_user' (3 interaction(s)):
  [2026-04-21 10:00:01] delaying     | I will pay later
  [2026-04-21 10:00:09] frustrated   | this is not working
  [2026-04-21 10:00:15] cooperative  | okay thanks
User: /quit
```

## CLI commands

| Command     | Description                                          |
| ----------- | ---------------------------------------------------- |
| `/history`  | Print all stored interactions for the current user.  |
| `/last`     | Print the most recent intent for the current user.   |
| `/help`     | Show available commands.                             |
| `/quit`     | Exit the REPL (Ctrl-C / Ctrl-D also work).           |

Anything that does not start with `/` is treated as a user utterance
and classified.

## Modules at a glance

### `agent.intent_classifier`

* `IntentClassifier.classify(message) -> IntentResult`
* Intents: `cooperative`, `delaying`, `frustrated`, `general`
* Scoring = weighted keyword hits + regex pattern hits, saturated to
  a `[0.0, 1.0]` confidence.

### `agent.memory_manager`

* `MemoryManager.initialize_db()`
* `MemoryManager.store_interaction(user_id, message, intent) -> int`
* `MemoryManager.get_user_history(user_id, limit=None) -> list[Interaction]`
* `MemoryManager.get_last_intent(user_id) -> str | None`

The SQLite schema auto-creates on first use and lives in
`agent_memory.db` by default.
