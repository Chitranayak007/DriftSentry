"""
DriftSentry ML / Behavioral Drift Engine
----------------------------------------
Prototype ML module designed to replace the hard-coded run_ml_analysis()
function in the existing FastAPI backend.

Model:
    Isolation Forest trained on synthetic "normal employee activity windows".

The model is intentionally combined with interpretable behavioral signals.
The ML model detects multivariate outliers; the signals explain WHY a
window looks unusual.

No user/security decision is made by the ML model alone.
"""

from __future__ import annotations

from datetime import datetime
from math import exp
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
N_TRAINING_WINDOWS = 1200
DECAY_LAMBDA = 0.08

# Resources that should be treated as sensitive in the prototype.
SENSITIVE_KEYWORDS = (
    "finance",
    "secret",
    "sensitive",
    "payroll",
    "credential",
    "prod-db",
    "admin-db",
)

# Known normal machines/resources/projects used by the synthetic prototype.
# These can later be replaced by learned per-user/per-team baselines.
NORMAL_MACHINES = {"dev-01", "dev-02", "dev-03"}
NORMAL_PROJECTS = {"Atlas", "Engineering"}
NORMAL_RESOURCE_PREFIXES = ("repo-",)


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

FEATURE_NAMES = [
    "event_count",
    "off_hours_ratio",
    "unique_machine_ratio",
    "unknown_machine_ratio",
    "unique_resource_ratio",
    "sensitive_resource_ratio",
    "download_ratio",
    "new_project_ratio",
    "night_access_ratio",
    "avg_hour_deviation",
]


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _is_off_hours(hour: int) -> bool:
    return hour < 7 or hour >= 19


def _is_night(hour: int) -> bool:
    return hour < 5


def _is_sensitive(resource: str) -> bool:
    value = resource.lower()
    return any(keyword in value for keyword in SENSITIVE_KEYWORDS)


def _event_feature_dict(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """Create interpretable aggregate features for one user's event window."""

    if not events:
        return {name: 0.0 for name in FEATURE_NAMES}

    parsed = [_parse_timestamp(e["timestamp"]) for e in events]

    hours = [dt.hour + dt.minute / 60.0 for dt in parsed]
    off_hours = [_is_off_hours(dt.hour) for dt in parsed]
    night = [_is_night(dt.hour) for dt in parsed]

    machines = [str(e.get("machine", "")) for e in events]
    resources = [str(e.get("resource", "")) for e in events]
    projects = [str(e.get("project", "")) for e in events]
    actions = [str(e.get("action", "")).upper() for e in events]

    unique_machine_ratio = len(set(machines)) / len(events)
    unknown_machine_ratio = sum(
        m.startswith("unknown") or m == "foreign-machine" for m in machines
    ) / len(events)

    unique_resource_ratio = len(set(resources)) / len(events)

    sensitive_ratio = sum(_is_sensitive(r) for r in resources) / len(events)
    download_ratio = sum(a in {"DOWNLOAD", "EXPORT", "COPY"} for a in actions) / len(events)

    # "New project" means outside the normal engineering project set.
    new_project_ratio = sum(
        project not in NORMAL_PROJECTS for project in projects
    ) / len(events)

    # Deviation from a normal 09:00–18:00 workday.
    hour_deviation = sum(
        0.0 if 9 <= hour <= 18 else min(abs(hour - 9), abs(hour - 18))
        for hour in hours
    ) / len(events)

    return {
        "event_count": float(len(events)),
        "off_hours_ratio": float(sum(off_hours) / len(events)),
        "unique_machine_ratio": float(unique_machine_ratio),
        "unknown_machine_ratio": float(unknown_machine_ratio),
        "unique_resource_ratio": float(unique_resource_ratio),
        "sensitive_resource_ratio": float(sensitive_ratio),
        "download_ratio": float(download_ratio),
        "new_project_ratio": float(new_project_ratio),
        "night_access_ratio": float(sum(night) / len(events)),
        "avg_hour_deviation": float(hour_deviation),
    }


def extract_features(events: List[Dict[str, Any]]) -> np.ndarray:
    features = _event_feature_dict(events)
    return np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)


# ---------------------------------------------------------------------
# Synthetic normal-data generator
# ---------------------------------------------------------------------

def _generate_normal_window(rng: np.random.Generator) -> List[Dict[str, Any]]:
    """
    Generate one synthetic normal employee activity window.

    This is training data for the hackathon prototype. In production,
    replace it with historical, labelled/validated enterprise telemetry.
    """
    count = int(rng.integers(3, 12))

    machines = ["dev-01", "dev-02", "dev-03"]
    resources = [
        "repo-atlas",
        "repo-engine",
        "repo-api",
        "repo-web",
        "repo-core",
    ]
    projects = ["Atlas", "Engineering"]

    events = []

    for i in range(count):
        day = int(rng.integers(1, 22))
        hour = int(rng.integers(9, 18))
        minute = int(rng.integers(0, 60))

        events.append(
            {
                "timestamp": f"2026-09-{day:02d}T{hour:02d}:{minute:02d}:00",
                "machine": str(rng.choice(machines)),
                "resource": str(rng.choice(resources)),
                "action": "READ",
                "project": str(rng.choice(projects)),
            }
        )

    return events


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model() -> IsolationForest:
    rng = np.random.default_rng(RANDOM_STATE)

    training_rows = []
    for _ in range(N_TRAINING_WINDOWS):
        events = _generate_normal_window(rng)
        feature_dict = _event_feature_dict(events)
        training_rows.append(
            [feature_dict[name] for name in FEATURE_NAMES]
        )

    X_train = np.asarray(training_rows, dtype=float)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


MODEL = build_model()


# ---------------------------------------------------------------------
# Explainable scoring
# ---------------------------------------------------------------------

def _manual_signal_score(features: Dict[str, float]) -> Tuple[float, List[str]]:
    """
    Deterministic interpretable score.

    This complements Isolation Forest. The ML model finds multivariate
    outliers; this layer identifies concrete reasons for the outlier.
    """

    score = 0.0
    signals: List[str] = []

    if features["off_hours_ratio"] > 0.25:
        score += 18
        signals.append("Unusual access time")

    if features["night_access_ratio"] > 0:
        score += 18
        signals.append("Night-time access")

    if features["unknown_machine_ratio"] > 0:
        score += 20
        signals.append("New or unknown machine")

    if features["sensitive_resource_ratio"] > 0:
        score += 25
        signals.append("Sensitive resource access")

    if features["download_ratio"] > 0:
        score += 15
        signals.append("Sensitive data download/export")

    if features["new_project_ratio"] > 0:
        score += 10
        signals.append("Access outside normal project scope")

    if features["unique_resource_ratio"] >= 0.75 and features["event_count"] >= 3:
        score += 8
        signals.append("High resource novelty")

    if features["unique_machine_ratio"] >= 0.75 and features["event_count"] >= 3:
        score += 6
        signals.append("Multiple machine changes")

    if features["avg_hour_deviation"] >= 4:
        score += 8
        signals.append("Large working-hour deviation")

    return min(score, 100.0), signals


def _ml_anomaly_score(features_array: np.ndarray) -> float:
    """
    Convert Isolation Forest's decision function into an intuitive 0–100
    anomaly score. Higher = more anomalous.
    """
    decision = float(MODEL.decision_function(features_array)[0])

    # IsolationForest decision values are normally positive for inliers and
    # negative for outliers. Map roughly into 0–100.
    score = 50.0 - (decision * 100.0)
    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------
# Decay-weighted cumulative risk
# ---------------------------------------------------------------------

def decay_weighted_risk(
    events: List[Dict[str, Any]],
    now: datetime | None = None,
) -> float:
    """
    R_total = sum(r_i * exp(-lambda * age_days))

    Each event receives an interpretable base risk and older events decay.
    """

    if not events:
        return 0.0

    if now is None:
        now = max(_parse_timestamp(e["timestamp"]) for e in events)

    total = 0.0

    for event in events:
        timestamp = _parse_timestamp(event["timestamp"])
        age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)

        resource = str(event.get("resource", ""))
        machine = str(event.get("machine", ""))
        action = str(event.get("action", "")).upper()
        hour = timestamp.hour

        event_risk = 5.0

        if _is_off_hours(hour):
            event_risk += 8.0

        if _is_night(hour):
            event_risk += 8.0

        if machine.startswith("unknown") or machine == "foreign-machine":
            event_risk += 12.0

        if _is_sensitive(resource):
            event_risk += 15.0

        if action in {"DOWNLOAD", "EXPORT", "COPY"}:
            event_risk += 10.0

        total += event_risk * exp(-DECAY_LAMBDA * age_days)

    # Keep the cumulative signal on a 0–100 scale for the API.
    return float(np.clip(total, 0.0, 100.0))


# ---------------------------------------------------------------------
# Main function used by FastAPI
# ---------------------------------------------------------------------

def run_ml_analysis(user_id: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Drop-in replacement for the backend's placeholder function.

    Returns the same core fields:
        anomaly_score
        signals

    plus additional fields that the frontend can optionally display.
    """

    features = _event_feature_dict(events)
    X = np.array(
        [[features[name] for name in FEATURE_NAMES]],
        dtype=float,
    )

    ml_score = _ml_anomaly_score(X)
    rule_score, signals = _manual_signal_score(features)
    cumulative_score = decay_weighted_risk(events)

    # ML gets the largest contribution, while the interpretable rules make
    # the result understandable and stable for a hackathon demo.
    combined = (
        0.55 * ml_score
        + 0.30 * rule_score
        + 0.15 * cumulative_score
    )

    # The anomaly_score is kept in the backend's expected 0–1 format.
    anomaly_score = float(np.clip(combined / 100.0, 0.0, 1.0))

    return {
        "anomaly_score": round(anomaly_score, 4),
        "ml_score": round(ml_score, 2),
        "behavioral_score": round(rule_score, 2),
        "cumulative_decay_score": round(cumulative_score, 2),
        "signals": signals,
        "features": {
            name: round(float(features[name]), 4)
            for name in FEATURE_NAMES
        },
        "model": "IsolationForest + explainable behavioral signals",
    }
