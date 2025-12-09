"""Runtime SHAP helper for IDS explanations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLAINER_PATH = PROJECT_ROOT / "models" / "shap_explainer.pkl"
TOP_FEATURES_PATH = PROJECT_ROOT / "models" / "top_20_features.json"

_EXPLAINER = None
_TOP_FEATURES = None


def _ensure_explainer():
    global _EXPLAINER
    if _EXPLAINER is None:
        if not EXPLAINER_PATH.exists():
            raise FileNotFoundError(f"SHAP explainer missing: {EXPLAINER_PATH}")
        _EXPLAINER = joblib.load(EXPLAINER_PATH)
    return _EXPLAINER


def _ensure_top_features():
    global _TOP_FEATURES
    if _TOP_FEATURES is None:
        if not TOP_FEATURES_PATH.exists():
            raise FileNotFoundError(f"Top feature list missing: {TOP_FEATURES_PATH}")
        with open(TOP_FEATURES_PATH, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        _TOP_FEATURES = set(payload.get("top_features", []))
    return _TOP_FEATURES


# Warm up on import for faster serving.
_ensure_explainer()
_ensure_top_features()


def explain_attack(
    input_vector: Sequence[float],
    feature_names: Iterable[str],
    attack_class_index: int = 1,
    top_n: int = 3,
) -> List[dict]:
    """Return the top-N contributing features among the curated top-20 list."""
    explainer = _ensure_explainer()
    top_features = _ensure_top_features()

    feature_names = list(feature_names)
    vector = np.asarray(input_vector, dtype=np.float64).reshape(1, -1)

    shap_values = explainer(vector)
    if isinstance(shap_values, list):
        attack_values = shap_values[attack_class_index][0]
    else:
        attack_values = shap_values[0]

    if len(feature_names) != len(attack_values):
        raise ValueError("Feature names count does not match SHAP contributions")

    contributions = []
    for name, value in zip(feature_names, attack_values):
        if name not in top_features:
            continue
        direction = "positive" if value >= 0 else "negative"
        magnitude = abs(float(value))
        contributions.append(
            {
                "feature": name,
                "direction": direction,
                "impact": magnitude,
                "description": f"{direction.capitalize()} impact (|SHAP|={magnitude:.4f})",
            }
        )

    contributions.sort(key=lambda item: item["impact"], reverse=True)
    formatted = []
    for entry in contributions[:top_n]:
        adjective = "High" if entry["impact"] >= 0.5 else "Medium" if entry["impact"] >= 0.2 else "Low"
        formatted.append(
            {
                "feature": entry["feature"],
                "impact": f"{adjective} ({entry['direction']} |SHAP|={entry['impact']:.4f})",
            }
        )
    return formatted


# --- Integration Guide ----------------------------------------------------
# live_bridge.py (önerilen kullanım):
# from utils.xai_engine import explain_attack
# ... tahmin döngüsünde ...
# reasons = explain_attack(X_scaled.iloc[idx], model_features.columns)
# log_attack(ip_addr, "BLOCKED", json.dumps(reasons)) veya dashboard'a gönderin.
