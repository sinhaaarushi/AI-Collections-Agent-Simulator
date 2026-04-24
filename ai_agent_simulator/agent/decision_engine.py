"""Rule-based decision layer over intent, history, and compliance score."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ai_agent_simulator.agent.intent_classifier import (
    COOPERATIVE,
    DELAYING,
    FRUSTRATED,
)


def decide(
    intent: str,
    user_profile: Mapping[str, Any],
    predicted_compliance: float,
) -> Dict[str, str]:
    """Choose the next agent action from intent and behavioral signals.

    Args:
        intent: Classified intent for the current user message.
        user_profile: Summary dict (same shape as for the behavior model).
        predicted_compliance: Probability in ``[0, 1]`` from the ML model.

    Returns:
        A dict with ``action`` and ``reason`` keys.
    """
    intent_norm = intent.strip().lower()
    delays = int(user_profile.get("num_delays", 0) or 0)
    frustrated_turns = int(user_profile.get("num_frustrated", 0) or 0)
    compliance = float(predicted_compliance)
    compliance = max(0.0, min(1.0, compliance))

    if intent_norm == FRUSTRATED and compliance < 0.4:
        return {
            "action": "escalate_to_human",
            "reason": _reason_escalate(frustrated_turns, compliance),
        }
    if intent_norm == DELAYING:
        return {
            "action": "send_reminder",
            "reason": _reason_reminder(delays, compliance),
        }
    if intent_norm == COOPERATIVE:
        return {
            "action": "assist_user",
            "reason": _reason_assist(compliance),
        }
    return {
        "action": "standard_response",
        "reason": _reason_standard(intent_norm, compliance),
    }


def _reason_escalate(frustrated_turns: int, compliance: float) -> str:
    parts = [
        "User shows frustration with low predicted compliance",
        f"(p={compliance:.2f})",
    ]
    if frustrated_turns >= 2:
        parts.insert(1, "and repeated frustrated turns")
    return " ".join(parts) + "; escalate_to_human reduces escalation risk."


def _reason_reminder(delays: int, compliance: float) -> str:
    if delays >= 2 and compliance < 0.4:
        return (
            "User shows repeated delay behavior with low compliance probability"
        )
    if delays >= 2:
        return (
            "User shows repeated delay behavior; a structured reminder is appropriate"
        )
    return (
        "User is deferring payment; send_reminder clarifies timelines and next steps"
    )


def _reason_assist(compliance: float) -> str:
    return (
        "User is cooperative"
        + (
            f" with favorable compliance outlook (p={compliance:.2f})"
            if compliance >= 0.5
            else "; assist_user keeps momentum despite softer compliance signal"
        )
    )


def _reason_standard(intent: str, compliance: float) -> str:
    return (
        f"Intent '{intent}' has no specialized branch; standard_response "
        f"with monitoring (compliance p={compliance:.2f})"
    )
