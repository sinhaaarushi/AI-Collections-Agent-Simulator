"""Logistic regression for payment compliance, trained on synthetic data only.

Maps :class:`~ai_agent_simulator.agent.profile_summary.UserProfileSummary`
plus last intent and a text-derived sentiment score into the feature dict
expected by :meth:`BehaviorModel.predict_compliance`.
"""

from __future__ import annotations

import math
import random
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ai_agent_simulator.agent.intent_classifier import ALL_INTENTS
from ai_agent_simulator.agent.profile_summary import UserProfileSummary

_INTENT_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ALL_INTENTS)}


def _generate_synthetic_dataset(
    n_samples: int = 800,
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build synthetic feature matrix X and binary compliance labels y.

    Features: num_interactions, num_delays, num_frustrated, last_intent index,
    sentiment_score. Labels come from a noisy logistic process (cooperative
    last intent and positive sentiment correlate with compliance).
    """
    rng = random.Random(seed)
    intents: Sequence[str] = list(ALL_INTENTS)

    rows_x: List[List[float]] = []
    rows_y: List[int] = []

    for _ in range(n_samples):
        num_interactions = rng.randint(1, 60)
        max_other = max(0, num_interactions - 1)
        num_delays = rng.randint(0, min(max_other, 25))
        remaining = max(0, max_other - num_delays)
        num_frustrated = rng.randint(0, min(remaining, 20))
        last_intent = rng.choice(intents)
        sentiment_score = rng.uniform(-1.0, 1.0)

        last_bias = {
            "cooperative": 1.15,
            "general": 0.15,
            "delaying": -0.65,
            "frustrated": -1.05,
        }[last_intent]

        z = (
            0.95 * sentiment_score
            + last_bias
            - 0.07 * num_delays
            - 0.09 * num_frustrated
            + 0.025 * min(num_interactions, 12)
            + rng.gauss(0.0, 0.45)
        )
        prob = 1.0 / (1.0 + math.exp(-z))
        compliance = 1 if rng.random() < prob else 0

        rows_x.append(
            [
                float(num_interactions),
                float(num_delays),
                float(num_frustrated),
                float(_INTENT_TO_INDEX.get(last_intent, _INTENT_TO_INDEX["general"])),
                float(sentiment_score),
            ]
        )
        rows_y.append(compliance)

    x = np.asarray(rows_x, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.int64)
    return x, y


def sentiment_score_from_text(message: str) -> float:
    """Rough sentiment in [-1, 1] from short positive/negative phrase hits."""
    if not message or not message.strip():
        return 0.0

    normalized = re.sub(r"\s+", " ", message.strip().lower())

    positive = (
        "thanks",
        "thank you",
        "okay",
        "great",
        "resolved",
        "appreciate",
        "will pay",
        "paid",
    )
    negative = (
        "not working",
        "issue",
        "problem",
        "bad",
        "frustrated",
        "angry",
        "stop",
        "harassment",
        "terrible",
        "hate",
    )

    pos_hits = sum(1 for p in positive if _phrase_hits(normalized, p))
    neg_hits = sum(1 for n in negative if _phrase_hits(normalized, n))
    raw = float(pos_hits - neg_hits)
    if raw == 0.0:
        return 0.0
    return max(-1.0, min(1.0, math.tanh(raw * 0.65)))


def _phrase_hits(normalized: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in normalized
    return re.search(rf"\b{re.escape(phrase)}\b", normalized) is not None


def build_user_profile_dict(
    summary: UserProfileSummary,
    *,
    last_intent: str,
    sentiment_score: float,
) -> Dict[str, float | int | str]:
    """Build the feature dict from SQLite-backed summary + current turn."""
    counts = summary.intent_counts
    return {
        "num_interactions": summary.interaction_count,
        "num_delays": int(counts.get("delaying", 0)),
        "num_frustrated": int(counts.get("frustrated", 0)),
        "last_intent": last_intent,
        "sentiment_score": float(sentiment_score),
    }


class BehaviorModel:
    """Train scaler + logistic regression on synthetic rows; predict compliance."""

    def __init__(self, *, n_samples: int = 800, random_seed: int = 42) -> None:
        x, y = _generate_synthetic_dataset(n_samples, seed=random_seed)
        self._scaler = StandardScaler()
        x_scaled = self._scaler.fit_transform(x)
        self._clf = LogisticRegression(max_iter=200, solver="lbfgs")
        self._clf.fit(x_scaled, y)

    def predict_compliance(self, user_profile: Mapping[str, object]) -> float:
        """Probability of compliance in [0, 1] (see :func:`build_user_profile_dict`)."""
        features = _profile_to_feature_row(user_profile)
        x = np.asarray([features], dtype=np.float64)
        x_scaled = self._scaler.transform(x)
        prob = float(self._clf.predict_proba(x_scaled)[0, 1])
        return max(0.0, min(1.0, prob))


def _profile_to_feature_row(user_profile: Mapping[str, object]) -> List[float]:
    last = str(user_profile.get("last_intent", "general")).strip().lower()
    intent_idx = float(_INTENT_TO_INDEX.get(last, _INTENT_TO_INDEX["general"]))

    def _to_float(key: str) -> float:
        val = user_profile.get(key, 0)
        if val is None:
            return 0.0
        return float(val)

    sentiment = float(user_profile.get("sentiment_score", 0.0))
    sentiment = max(-1.0, min(1.0, sentiment))

    return [
        _to_float("num_interactions"),
        _to_float("num_delays"),
        _to_float("num_frustrated"),
        intent_idx,
        sentiment,
    ]


def predict_compliance(
    user_profile: Mapping[str, object],
    *,
    model: Optional[BehaviorModel] = None,
) -> float:
    """Convenience wrapper; pass ``model`` to avoid retraining on every call."""
    m = model if model is not None else BehaviorModel()
    return m.predict_compliance(user_profile)
