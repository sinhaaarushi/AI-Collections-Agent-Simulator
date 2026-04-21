"""Command-line entry point for the AI Collections Agent Simulator.

Running this module starts an interactive REPL: the user types a
message, the agent classifies the intent, persists the interaction to
SQLite, and prints a structured result.

Usage::

    python -m ai_agent_simulator.app
    python -m ai_agent_simulator.app --user-id alice --db ./memory.db

Type ``/history`` to see the stored history for the current user,
``/last`` to print the last detected intent, or ``/quit`` (or Ctrl-D /
Ctrl-C) to exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, TextIO

from ai_agent_simulator.agent.intent_classifier import IntentClassifier, IntentResult
from ai_agent_simulator.agent.memory_manager import MemoryManager

DEFAULT_USER_ID = "default_user"
DEFAULT_DB_PATH = Path("agent_memory.db")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ai-collections-agent",
        description="Local AI Collections Agent Simulator (Day 1 foundation).",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help=f"Identifier for the conversational user (default: {DEFAULT_USER_ID!r}).",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to the SQLite database file (default: {str(DEFAULT_DB_PATH)!r}).",
    )
    return parser.parse_args(argv)


def _format_result(result: IntentResult) -> str:
    """Format an :class:`IntentResult` for CLI output."""
    return f"Detected Intent: {result.intent} (confidence: {result.confidence:.2f})"


def _print_history(memory: MemoryManager, user_id: str, out: TextIO) -> None:
    """Print the stored history for ``user_id`` to ``out``."""
    history = memory.get_user_history(user_id)
    if not history:
        print(f"(no history for user {user_id!r})", file=out)
        return
    print(f"History for {user_id!r} ({len(history)} interaction(s)):", file=out)
    for item in history:
        ts = item.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{ts}] {item.intent:<12} | {item.message}", file=out)


def _print_last_intent(memory: MemoryManager, user_id: str, out: TextIO) -> None:
    """Print the last intent recorded for ``user_id``."""
    last = memory.get_last_intent(user_id)
    if last is None:
        print(f"(no prior intent for user {user_id!r})", file=out)
    else:
        print(f"Last intent for {user_id!r}: {last}", file=out)


def _handle_command(
    command: str,
    memory: MemoryManager,
    user_id: str,
    out: TextIO,
) -> bool:
    """Handle a slash command. Returns ``True`` if the REPL should exit."""
    cmd = command.strip().lower()
    if cmd in {"/quit", "/exit"}:
        return True
    if cmd == "/history":
        _print_history(memory, user_id, out)
    elif cmd == "/last":
        _print_last_intent(memory, user_id, out)
    elif cmd in {"/help", "/?"}:
        print(
            "Commands: /history, /last, /quit. Anything else is classified.",
            file=out,
        )
    else:
        print(f"Unknown command: {command!r}. Type /help for options.", file=out)
    return False


def run_repl(
    classifier: IntentClassifier,
    memory: MemoryManager,
    user_id: str,
    *,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
) -> None:
    """Run the interactive classification loop until EOF or ``/quit``.

    Args:
        classifier: The intent classifier to use.
        memory: The memory manager used to persist interactions.
        user_id: Identifier for the current conversational user.
        stdin: Input stream. Defaults to :data:`sys.stdin`.
        stdout: Output stream. Defaults to :data:`sys.stdout`.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    memory.initialize_db()

    print(
        f"AI Collections Agent Simulator (user: {user_id!r}). "
        "Type /help for commands, /quit to exit.",
        file=stdout,
    )

    while True:
        try:
            stdout.write("User: ")
            stdout.flush()
            line = stdin.readline()
        except KeyboardInterrupt:
            print("\nExiting.", file=stdout)
            return

        if not line:
            print("\nExiting.", file=stdout)
            return

        message = line.rstrip("\r\n")
        if not message.strip():
            continue

        if message.startswith("/"):
            if _handle_command(message, memory, user_id, stdout):
                return
            continue

        result = classifier.classify(message)
        memory.store_interaction(user_id, message, result.intent)
        print(_format_result(result), file=stdout)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)
    classifier = IntentClassifier()
    memory = MemoryManager(db_path=args.db)
    try:
        run_repl(classifier, memory, args.user_id)
    except KeyboardInterrupt:
        print("\nExiting.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
