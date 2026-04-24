"""Command-line entry point for the AI Collections Agent Simulator.

Running this module starts an interactive REPL: the user types a
message, the agent classifies intent, stores the turn in SQLite, estimates
compliance with a local logistic model, runs the decision engine, and prints
action and reasoning.

Usage::

    python -m ai_agent_simulator.app
    python -m ai_agent_simulator.app --user-id alice --db ./memory.db

Type ``/history`` to see stored turns, ``/profile`` for an intent
summary, ``/last`` for the latest intent, or ``/quit`` (or Ctrl-D /
Ctrl-C) to exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, TextIO

from ai_agent_simulator.agent.behavior_model import (
    BehaviorModel,
    build_user_profile_dict,
    sentiment_score_from_text,
)
from ai_agent_simulator.agent.decision_engine import decide
from ai_agent_simulator.agent.intent_classifier import IntentClassifier, IntentResult
from ai_agent_simulator.agent.memory_manager import MemoryManager
from ai_agent_simulator.agent.profile_summary import (
    build_profile_summary,
    format_profile_summary_text,
)

DEFAULT_USER_ID = "default_user"
DEFAULT_DB_PATH = Path("agent_memory.db")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ai-collections-agent",
        description="Local AI Collections Agent Simulator (Day 2: ML + decisions).",
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


def _print_debug_classification(result: IntentResult, out: TextIO) -> None:
    """Print a one-line trace-style summary after intent classification."""
    print(
        f"[AGENT DEBUG] Intent classified as: {result.intent} "
        f"with confidence {result.confidence:.2f}",
        file=out,
    )


def _print_decision_trace(
    profile: dict[str, object],
    compliance: float,
    action: str,
    out: TextIO,
) -> None:
    """Print profile counts and outcome for quick operator visibility."""
    delays = profile.get("num_delays", 0)
    frustrated = profile.get("num_frustrated", 0)
    print(
        f"[AGENT TRACE] history delays={delays} frustrated={frustrated} "
        f"compliance={compliance:.2f} -> action={action}",
        file=out,
    )


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


def _print_profile(memory: MemoryManager, user_id: str, out: TextIO) -> None:
    """Print a profile summary derived from this user's interaction history."""
    history = memory.get_user_history(user_id)
    summary = build_profile_summary(user_id, history)
    print(format_profile_summary_text(summary), file=out)


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
    elif cmd == "/profile":
        _print_profile(memory, user_id, out)
    elif cmd == "/last":
        _print_last_intent(memory, user_id, out)
    elif cmd in {"/help", "/?"}:
        print(
            "Commands: /history, /profile, /last, /quit. "
            "Anything else is classified.",
            file=out,
        )
    else:
        print(f"Unknown command: {command!r}. Type /help for options.", file=out)
    return False


def run_repl(
    classifier: IntentClassifier,
    memory: MemoryManager,
    behavior_model: BehaviorModel,
    user_id: str,
    *,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
) -> None:
    """Run the interactive classification loop until EOF or ``/quit``.

    Args:
        classifier: The intent classifier to use.
        memory: The memory manager used to persist interactions.
        behavior_model: Trained compliance model (synthetic training data).
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
            print(
                "(empty line - type a message or /help)",
                file=stdout,
            )
            continue

        if message.startswith("/"):
            if _handle_command(message, memory, user_id, stdout):
                return
            continue

        result = classifier.classify(message)
        _print_debug_classification(result, stdout)
        memory.store_interaction(user_id, message, result.intent)

        history = memory.get_user_history(user_id)
        summary = build_profile_summary(user_id, history)
        last_intent = memory.get_last_intent(user_id) or result.intent
        sentiment = sentiment_score_from_text(message)
        profile = build_user_profile_dict(
            summary,
            last_intent=last_intent,
            sentiment_score=sentiment,
        )
        compliance_prob = behavior_model.predict_compliance(profile)
        decision = decide(result.intent, profile, compliance_prob)
        _print_decision_trace(profile, compliance_prob, decision["action"], stdout)

        print(f"Intent: {result.intent}", file=stdout)
        print(f"Compliance Probability: {compliance_prob:.2f}", file=stdout)
        print(f"Action: {decision['action']}", file=stdout)
        print(f"Reason: {decision['reason']}", file=stdout)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)
    classifier = IntentClassifier()
    memory = MemoryManager(db_path=args.db)
    behavior_model = BehaviorModel()
    try:
        run_repl(classifier, memory, behavior_model, args.user_id)
    except KeyboardInterrupt:
        print("\nExiting.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
