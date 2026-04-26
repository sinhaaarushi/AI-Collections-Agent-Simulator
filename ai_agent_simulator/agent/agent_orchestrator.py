"""End-to-end orchestration for the local collections agent pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ai_agent_simulator.agent.action_handler import (
    execute_action,
    simulate_action_outcome,
)
from ai_agent_simulator.agent.behavior_model import (
    BehaviorModel,
    build_user_profile_dict,
    sentiment_score_from_text,
)
from ai_agent_simulator.agent.decision_engine import decide
from ai_agent_simulator.agent.intent_classifier import IntentClassifier
from ai_agent_simulator.agent.memory_manager import MemoryManager
from ai_agent_simulator.agent.profile_summary import build_profile_summary

# Below this, intent labels are treated as uncertain and get a safe default path.
_LOW_INTENT_CONFIDENCE_THRESHOLD = 0.35


class AgentOrchestrator:
    """Coordinate intent, memory, prediction, decisions, and actions."""

    def __init__(
        self,
        *,
        classifier: Optional[IntentClassifier] = None,
        memory: Optional[MemoryManager] = None,
        behavior_model: Optional[BehaviorModel] = None,
    ) -> None:
        """Create an orchestrator with injectable components for CLI/tests."""
        self._classifier = classifier or IntentClassifier()
        self._memory = memory or MemoryManager()
        self._behavior_model = behavior_model or BehaviorModel()

    def run(self, user_id: str, message: str) -> Dict[str, Any]:
        """Run the full local agent pipeline for a single user message.

        Flow:
            1. Classify intent.
            2. Store the interaction.
            3. Build the user's memory-backed profile.
            4. Predict compliance probability.
            5. Decide the next action.
            6. Execute the simulated action.
        """
        normalized_user_id = user_id.strip() or "default_user"
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("message must be a non-empty string")

        self._memory.initialize_db()

        result = self._classifier.classify(normalized_message)
        self._memory.store_interaction(
            normalized_user_id,
            normalized_message,
            result.intent,
        )

        profile = self._build_profile(normalized_user_id, normalized_message, result.intent)
        compliance_probability = self._behavior_model.predict_compliance(profile)
        if result.confidence < _LOW_INTENT_CONFIDENCE_THRESHOLD:
            decision = {
                "action": "standard_response",
                "reason": (
                    "Classifier confidence is low; using a safe standard response "
                    "instead of a specialized path"
                ),
                "decision_confidence": 0.55,
            }
        else:
            decision = decide(result.intent, profile, compliance_probability)
        action = str(decision["action"])
        response = execute_action(action, normalized_user_id)
        strategy = _strategy_for_action(action, str(profile["current_strategy"]))
        outcome = simulate_action_outcome(action)
        self._memory.update_current_strategy(normalized_user_id, strategy)
        self._memory.update_action_outcome(normalized_user_id, outcome)
        self._memory.record_action_metric(action)
        metrics = self._memory.get_action_metrics()

        return {
            "intent": result.intent,
            "confidence": round(result.confidence, 2),
            "compliance_probability": round(compliance_probability, 2),
            "decision_confidence": round(float(decision["decision_confidence"]), 2),
            "action": action,
            "reason": str(decision["reason"]),
            "response": response,
            "outcome": outcome,
            "strategy": strategy,
            "metrics": metrics,
        }

    def get_decision_factors(
        self,
        user_id: str,
        agent_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build trace-friendly factors for CLI and UI display."""
        normalized_user_id = user_id.strip() or "default_user"
        self._memory.initialize_db()
        history = self._memory.get_user_history(normalized_user_id)
        summary = build_profile_summary(normalized_user_id, history)

        return {
            "intent": agent_result["intent"],
            "num_interactions": summary.interaction_count,
            "num_delays": int(summary.intent_counts.get("delaying", 0)),
            "num_frustrated": int(summary.intent_counts.get("frustrated", 0)),
            "compliance": agent_result["compliance_probability"],
            "strategy": agent_result["strategy"],
            "outcome": agent_result["outcome"],
        }

    def _build_profile(
        self,
        user_id: str,
        message: str,
        last_intent: str,
    ) -> Dict[str, float | int | str]:
        """Create the behavior-model profile for the latest interaction."""
        history = self._memory.get_user_history(user_id)
        summary = build_profile_summary(user_id, history)
        sentiment = sentiment_score_from_text(message)
        profile = build_user_profile_dict(
            summary,
            last_intent=last_intent,
            sentiment_score=sentiment,
        )
        profile["current_strategy"] = self._memory.get_current_strategy(user_id)
        return profile


def _strategy_for_action(action: str, current_strategy: str) -> str:
    """Map the selected action to the persisted strategy state."""
    if action == "send_reminder_soft":
        return "soft_reminder"
    if action == "send_reminder_firm":
        return "firm_reminder"
    if action == "escalate_to_human":
        return "escalation"
    return current_strategy


_DEFAULT_ORCHESTRATOR = AgentOrchestrator()


def run_agent(user_id: str, message: str) -> Dict[str, Any]:
    """Run the default orchestrator for a single UI/demo message."""
    return _DEFAULT_ORCHESTRATOR.run(user_id, message)


def get_decision_factors(user_id: str, agent_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build trace factors from the default orchestrator."""
    return _DEFAULT_ORCHESTRATOR.get_decision_factors(user_id, agent_result)
