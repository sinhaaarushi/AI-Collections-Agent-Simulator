"""Agent package for the AI Collections Agent Simulator.

Exposes the intent classifier, memory manager, and profile helpers as the
main building blocks of the agent's runtime.
"""

from ai_agent_simulator.agent.intent_classifier import (
    IntentClassifier,
    IntentResult,
)
from ai_agent_simulator.agent.memory_manager import MemoryManager
from ai_agent_simulator.agent.profile_summary import (
    UserProfileSummary,
    build_profile_summary,
    format_profile_summary_text,
)

__all__ = [
    "IntentClassifier",
    "IntentResult",
    "MemoryManager",
    "UserProfileSummary",
    "build_profile_summary",
    "format_profile_summary_text",
]
