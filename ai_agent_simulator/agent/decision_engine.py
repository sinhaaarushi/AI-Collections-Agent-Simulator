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
    if frustrated_turns >= 2:
        return (
            f"Several frustrated touches plus weak compliance (p={compliance:.2f}); "
            "better to get a human on the line before things get worse"
        )
    return (
        f"Frustrated and the model only sees p={compliance:.2f} compliance; "
        "escalate before the tone hardens"
    )


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
        "Deferring payment this time; a reminder with a firm date usually helps"
    )


def _reason_assist(compliance: float) -> str:
    if compliance >= 0.5:
        return (
            f"Cooperative and compliance looks decent (p={compliance:.2f}); "
            "keep the conversation helpful"
        )
    return (
        "Cooperative tone but compliance estimate is soft; stay helpful and "
        "nudge toward a concrete commitment"
    )


def _reason_standard(intent: str, compliance: float) -> str:
    return (
        f"Nothing special for '{intent}' right now; answer normally and "
        f"watch compliance (p={compliance:.2f})"
    )
