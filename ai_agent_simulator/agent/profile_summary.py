"""Build a concise user profile from stored interaction history.

Pure functions over :class:`~ai_agent_simulator.agent.memory_manager.Interaction`
records: no extra database queries beyond whatever history the caller loads.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from ai_agent_simulator.agent.memory_manager import Interaction


@dataclass(frozen=True)
class UserProfileSummary:
    """Aggregated view of one user's past intents and activity.

    Attributes:
        user_id: Same identifier used with :class:`MemoryManager`.
        interaction_count: Number of stored messages for this user.
        intent_counts: Counts per intent label (e.g. ``cooperative`` -> 3).
        first_interaction_at: Timestamp of earliest message, if any.
        last_interaction_at: Timestamp of latest message, if any.
        dominant_intent: Intent with highest count; ties broken by most
            recent occurrence in the timeline.
    """

    user_id: str
    interaction_count: int
    intent_counts: Dict[str, int]
    first_interaction_at: Optional[datetime]
    last_interaction_at: Optional[datetime]
    dominant_intent: Optional[str]


def build_profile_summary(
    user_id: str,
    history: Sequence[Interaction],
) -> UserProfileSummary:
    """Compute a :class:`UserProfileSummary` from an in-memory history list.

    Args:
        user_id: User identifier (echoed in the result).
        history: Chronological or unsorted list; ordering is only used for
            first/last timestamps and tie-breaking.

    Returns:
        A summary with zeroed fields when ``history`` is empty.
    """
    rows: List[Interaction] = list(history)
    if not rows:
        return UserProfileSummary(
            user_id=user_id,
            interaction_count=0,
            intent_counts={},
            first_interaction_at=None,
            last_interaction_at=None,
            dominant_intent=None,
        )

    by_time = sorted(rows, key=lambda r: (r.timestamp, r.id))
    first, last = by_time[0], by_time[-1]
    counts = Counter(r.intent for r in rows)
    dominant = _resolve_dominant_intent(by_time, counts)

    return UserProfileSummary(
        user_id=user_id,
        interaction_count=len(rows),
        intent_counts=dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        first_interaction_at=first.timestamp,
        last_interaction_at=last.timestamp,
        dominant_intent=dominant,
    )


def _resolve_dominant_intent(
    sorted_by_time: Sequence[Interaction],
    counts: Counter,
) -> Optional[str]:
    """Pick the top intent; on ties, prefer the one seen most recently."""
    if not counts:
        return None
    top = max(counts.values())
    candidates = {k for k, v in counts.items() if v == top}
    if len(candidates) == 1:
        return next(iter(candidates))
    for interaction in reversed(sorted_by_time):
        if interaction.intent in candidates:
            return interaction.intent
    return next(iter(candidates))


def format_profile_summary_text(summary: UserProfileSummary) -> str:
    """Return a short multi-line block suitable for printing to the CLI.

    Args:
        summary: Profile to render.

    Returns:
        Human-readable text without a trailing newline beyond the last line.
    """
    lines: List[str] = [
        f"Profile: {summary.user_id!r}",
        f"  Total interactions: {summary.interaction_count}",
    ]
    if summary.interaction_count == 0:
        lines.append("  No stored messages yet. Send at least one utterance.")
        return "\n".join(lines)

    first = summary.first_interaction_at
    last = summary.last_interaction_at
    if first is not None:
        lines.append(f"  First message: {first:%Y-%m-%d %H:%M:%S}")
    if last is not None:
        lines.append(f"  Last message:  {last:%Y-%m-%d %H:%M:%S}")
    lines.append("  Intent counts:")
    for intent, n in summary.intent_counts.items():
        lines.append(f"    {intent}: {n}")
    lines.append(f"  Dominant intent: {summary.dominant_intent}")
    return "\n".join(lines)
