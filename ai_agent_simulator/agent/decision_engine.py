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
) -> Dict[str, object]:
    """Choose the next agent action from intent and behavioral signals.

    Args:
        intent: Classified intent for the current user message.
        user_profile: Summary dict (same shape as for the behavior model).
        predicted_compliance: Probability in ``[0, 1]`` from the ML model.

    Returns:
        A dict with ``action``, ``reason``, and ``decision_confidence`` keys.
    """
    intent_norm = intent.strip().lower()
    delays = int(user_profile.get("num_delays", 0) or 0)
    frustrated_turns = int(user_profile.get("num_frustrated", 0) or 0)
    current_strategy = str(user_profile.get("current_strategy", "soft_reminder"))
    compliance = float(predicted_compliance)
    compliance = max(0.0, min(1.0, compliance))

    if intent_norm == FRUSTRATED and compliance < 0.4:
        return {
            "action": "escalate_to_human",
            "reason": _reason_escalate(frustrated_turns, compliance),
            "decision_confidence": 0.94,
        }
    if intent_norm == DELAYING:
        action = _action_for_delay_count(delays)
        return {
            "action": action,
            "reason": _reason_for_delay_strategy(delays, current_strategy),
            "decision_confidence": _confidence_for_delay(delays, compliance, action),
        }
    if intent_norm == COOPERATIVE:
        return {
            "action": "assist_user",
            "reason": _reason_assist(compliance),
            "decision_confidence": 0.86,
        }
    return {
        "action": "standard_response",
        "reason": _reason_standard(intent_norm, compliance),
        "decision_confidence": 0.48,
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


def _action_for_delay_count(delays: int) -> str:
    """Map repeated delay behavior to the next action."""
    if delays <= 1:
        return "send_reminder_soft"
    if delays == 2:
        return "send_reminder_firm"
    return "escalate_to_human"


def _reason_for_delay_strategy(delays: int, current_strategy: str) -> str:
    """Explain how delay history changes the agent's strategy."""
    if delays <= 1:
        return (
            "First delay detected; starting with a soft reminder to keep the "
            "conversation cooperative"
        )
    if delays == 2:
        return (
            "User delayed again; escalating strategy from soft reminder to "
            "firm reminder"
        )
    return (
        "User has shown repeated delay behavior across multiple interactions; "
        f"escalating strategy from {current_strategy} to escalation"
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


def _confidence_for_delay(delays: int, compliance: float, action: str) -> float:
    """Return a simple confidence score for delay-handling decisions."""
    if action == "escalate_to_human":
        return 0.94 if compliance < 0.5 else 0.9
    if delays == 2:
        return 0.86
    return 0.78
