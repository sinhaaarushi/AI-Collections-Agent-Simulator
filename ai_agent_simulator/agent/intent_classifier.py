"""Rule-based intent classifier for the collections agent.

The classifier maps a raw user utterance to one of four intents used
throughout the simulator:

* ``cooperative`` -- the user is agreeing, thanking, or confirming.
* ``delaying``    -- the user is deferring payment or asking for more time.
* ``frustrated``  -- the user is expressing anger, complaints, or distress.
* ``general``     -- fallback bucket for anything that doesn't match.

The implementation is intentionally simple: a weighted keyword/phrase
lookup first; regular-expression patterns come in the next commit. It favours readability and extensibility over raw accuracy, and new keyword rows can be added without touching the scoring loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Pattern, Tuple

Intent = str

COOPERATIVE: Intent = "cooperative"
DELAYING: Intent = "delaying"
FRUSTRATED: Intent = "frustrated"
GENERAL: Intent = "general"

ALL_INTENTS: Tuple[Intent, ...] = (COOPERATIVE, DELAYING, FRUSTRATED, GENERAL)


# ---------------------------------------------------------------------------
# Keyword / phrase dictionaries
# ---------------------------------------------------------------------------
#
# Each intent has an ordered list of (phrase, weight) tuples. Weights are
# summed for every matching phrase to produce a raw score per intent.
# Multi-word phrases are matched as substrings; single words are matched
# on word boundaries so that, e.g., "payable" does not trigger "pay".

_KEYWORDS: Dict[Intent, List[Tuple[str, float]]] = {
    COOPERATIVE: [
        ("okay thanks", 2.0),
        ("thank you", 2.0),
        ("thanks", 1.5),
        ("sure", 1.2),
        ("sounds good", 1.5),
        ("will do", 1.5),
        ("i agree", 1.5),
        ("agreed", 1.2),
        ("no problem", 1.2),
        ("got it", 1.2),
        ("understood", 1.2),
        ("paid", 1.5),
        ("already paid", 2.0),
        ("ok", 0.8),
        ("okay", 1.0),
        ("yes", 0.8),
        ("alright", 1.0),
    ],
    DELAYING: [
        ("pay later", 2.5),
        ("pay next", 2.0),
        ("pay tomorrow", 2.0),
        ("pay next week", 2.5),
        ("pay next month", 2.5),
        ("need more time", 2.5),
        ("give me time", 2.0),
        ("some time", 1.2),
        ("extension", 2.0),
        ("postpone", 2.0),
        ("reschedule", 2.0),
        ("after", 0.6),
        ("later", 1.2),
        ("tomorrow", 1.0),
        ("next week", 1.5),
        ("next month", 1.5),
        ("not right now", 1.5),
        ("busy", 0.8),
        ("salary", 1.0),
        ("payday", 1.2),
    ],
    FRUSTRATED: [
        ("not working", 2.0),
        ("doesn't work", 2.0),
        ("does not work", 2.0),
        ("stop calling", 2.5),
        ("leave me alone", 2.5),
        ("fed up", 2.5),
        ("sick of", 2.0),
        ("ridiculous", 2.0),
        ("unacceptable", 2.0),
        ("terrible", 1.5),
        ("worst", 1.5),
        ("hate", 1.5),
        ("angry", 1.5),
        ("annoyed", 1.5),
        ("frustrated", 2.0),
        ("useless", 1.5),
        ("scam", 2.0),
        ("harassment", 2.5),
        ("complaint", 1.5),
        ("complain", 1.2),
        ("damn", 1.0),
        ("wtf", 1.5),
    ],
}


# Regex patterns are introduced in the following commit; the scorer still calls the same hook with an empty map.
_PATTERNS: Dict[Intent, List[Tuple[Pattern[str], float]]] = {}


@dataclass(frozen=True)
class IntentResult:
    """Structured result returned by :class:`IntentClassifier`.

    Attributes:
        intent: The predicted intent label.
        confidence: A value in ``[0.0, 1.0]`` describing how confident the
            classifier is in its prediction. For the fallback ``general``
            intent the confidence is a small constant.
    """

    intent: Intent
    confidence: float

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable representation of the result."""
        return {"intent": self.intent, "confidence": round(self.confidence, 2)}


class IntentClassifier:
    """Rule-based classifier that scores an utterance across intents.

    The classifier is deterministic and stateless; a single instance can
    be reused across many calls. Scoring works as follows:

    1. Normalize the input (lowercase, collapse whitespace).
    2. For each intent, sum the weights of all matching keyword and phrase hits (regex hooks are wired but empty until the next commit).
    3. Pick the intent with the highest score. Ties and zero scores fall
       back to :data:`GENERAL`.
    4. Map the winning raw score to a confidence in ``[0.0, 1.0]`` using
       a simple saturating transform.
    """

    #: Raw score at which confidence saturates to 1.0. Tuned empirically
    #: against the examples in the specification.
    _CONFIDENCE_SATURATION: float = 3.0

    #: Confidence assigned to the fallback ``general`` intent.
    _GENERAL_CONFIDENCE: float = 0.3

    def __init__(
        self,
        keywords: Dict[Intent, List[Tuple[str, float]]] | None = None,
        patterns: Dict[Intent, List[Tuple[Pattern[str], float]]] | None = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            keywords: Optional override for the keyword map. Defaults to
                the module-level :data:`_KEYWORDS`.
            patterns: Optional override for the regex pattern map.
                Defaults to :data:`_PATTERNS`.
        """
        self._keywords = keywords if keywords is not None else _KEYWORDS
        self._patterns = patterns if patterns is not None else _PATTERNS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, message: str) -> IntentResult:
        """Classify ``message`` and return an :class:`IntentResult`.

        Args:
            message: Raw user utterance.

        Returns:
            An :class:`IntentResult` with the predicted intent and a
            confidence score in ``[0.0, 1.0]``.
        """
        if not message or not message.strip():
            return IntentResult(intent=GENERAL, confidence=self._GENERAL_CONFIDENCE)

        normalized = self._normalize(message)
        scores = self._score_all(normalized)

        best_intent, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score <= 0.0:
            return IntentResult(intent=GENERAL, confidence=self._GENERAL_CONFIDENCE)

        confidence = min(1.0, best_score / self._CONFIDENCE_SATURATION)
        return IntentResult(intent=best_intent, confidence=confidence)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(message: str) -> str:
        """Lowercase and collapse whitespace for consistent matching."""
        return re.sub(r"\s+", " ", message.strip().lower())

    def _score_all(self, normalized: str) -> Dict[Intent, float]:
        """Compute the raw score for every non-fallback intent."""
        scores: Dict[Intent, float] = {
            intent: 0.0 for intent in ALL_INTENTS if intent != GENERAL
        }
        for intent in scores:
            scores[intent] += self._score_keywords(
                normalized, self._keywords.get(intent, [])
            )
            scores[intent] += self._score_patterns(
                normalized, self._patterns.get(intent, [])
            )
        return scores

    @staticmethod
    def _score_keywords(
        normalized: str, entries: Iterable[Tuple[str, float]]
    ) -> float:
        """Sum the weights of all matching keywords or phrases."""
        total = 0.0
        for phrase, weight in entries:
            phrase_norm = phrase.lower()
            if " " in phrase_norm:
                if phrase_norm in normalized:
                    total += weight
            else:
                # Word-boundary match for single tokens to avoid partial hits.
                if re.search(rf"\b{re.escape(phrase_norm)}\b", normalized):
                    total += weight
        return total

    @staticmethod
    def _score_patterns(
        normalized: str, entries: Iterable[Tuple[Pattern[str], float]]
    ) -> float:
        """Sum the weights of all matching regex patterns."""
        total = 0.0
        for pattern, weight in entries:
            if pattern.search(normalized):
                total += weight
        return total


def classify_intent(message: str) -> IntentResult:
    """Module-level convenience wrapper around :class:`IntentClassifier`.

    A fresh classifier is cheap to instantiate, but callers running in a
    tight loop should prefer reusing a single :class:`IntentClassifier`
    instance.
    """
    return IntentClassifier().classify(message)
