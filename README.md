# AI Collections Agent Simulator

A local, rule-based simulator of a financial collections/support agent.
It classifies user utterances into intents, stores the interaction
history in SQLite, derives a small **user profile** from that history,
and exposes a CLI for trying the flow end-to-end.

Everything runs offline: Python standard library only (no cloud APIs, no ML).

## First phase capabilities

| Area | What you get |
| ---- | ------------ |
| **Intent** | Four labels: `cooperative`, `delaying`, `frustrated`, `general`. Keyword and regex scoring, plus light sentiment hints (positive phrasing nudges cooperative; negative nudges frustrated). |
| **Confidence** | Reported scores are soft-capped (not full 1.0) so labels feel plausible, not overconfident. |
| **Memory** | SQLite persistence: each turn stores `user_id`, message, predicted intent, timestamp. |
| **CLI** | Read-eval-print loop, slash commands for history and profile, optional `[AGENT DEBUG]` line after classification. |
| **Profile** | `/profile` summarizes total turns, per-intent counts, first/last message times, and a dominant intent (ties broken by recency). |

## Project layout

```
ai_agent_simulator/
    __init__.py
    app.py                       # CLI entry point
    agent/
        __init__.py
        intent_classifier.py     # Rule-based intent + sentiment hints
        memory_manager.py        # SQLite-backed interaction memory
        profile_summary.py       # Profile aggregates from history
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
[AGENT DEBUG] Intent classified as: delaying with confidence 0.90
Detected Intent: delaying (confidence: 0.90)
User: /profile
Profile: 'default_user'
  Total interactions: 1
  First message: 2026-04-21 10:00:01
  Last message:  2026-04-21 10:00:01
  Intent counts:
    delaying: 1
  Dominant intent: delaying
User: /quit
```

Empty lines print a short hint instead of failing silently.

## CLI commands

| Command | Description |
| ------- | ----------- |
| `/history` | Print all stored interactions for the current user. |
| `/profile` | Print intent counts, dominant intent, and first/last message times. |
| `/last` | Print the most recent intent for the current user. |
| `/help` | List commands. |
| `/quit` | Exit the REPL (Ctrl-C and Ctrl-D also work). |

Anything that does not start with `/` is treated as a user utterance
and classified.

## Modules at a glance

### `agent.intent_classifier`

* `IntentClassifier.classify(message) -> IntentResult`
* Intents: `cooperative`, `delaying`, `frustrated`, `general`
* Weighted keywords, regex patterns, sentiment phrase boosts; confidence is capped below 1.0.

### `agent.memory_manager`

* `MemoryManager.initialize_db()`
* `MemoryManager.store_interaction(user_id, message, intent) -> int`
* `MemoryManager.get_user_history(user_id, limit=None) -> list[Interaction]`
* `MemoryManager.get_last_intent(user_id) -> str | None`

The SQLite schema auto-creates on first use. Default DB filename:
`agent_memory.db` (override with `--db`).

### `agent.profile_summary`

* `build_profile_summary(user_id, history) -> UserProfileSummary`
* `format_profile_summary_text(summary) -> str`
