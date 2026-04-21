"""Agent package for the AI Collections Agent Simulator.

Exposes the intent classifier as a primary building block of the
agent's runtime.
"""

from ai_agent_simulator.agent.intent_classifier import (
    IntentClassifier,
    IntentResult,
)

__all__ = ["IntentClassifier", "IntentResult"]
