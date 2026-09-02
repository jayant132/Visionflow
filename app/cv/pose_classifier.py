"""Keras MLP classifier over joint-angle features, with graceful fallback
to the deterministic rule engine if no trained model is present (so the
app always works out-of-the-box, even before running scripts/train_model.py).
"""
import os
import numpy as np

from app.agent.pose_rules import ASANA_RULES, check_pose
from app.telemetry import log

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pose_classifier.keras")
ASANAS = list(ASANA_RULES.keys())
FEATURE_JOINTS = ["left_elbow", "right_elbow", "left_knee", "right_knee", "left_hip", "right_hip"]

_model = None
_model_loaded_attempted = False


def _angles_to_vector(angles: dict) -> np.ndarray:
    return np.array([[angles.get(j, 0.0) / 180.0 for j in FEATURE_JOINTS]], dtype="float32")


def _try_load_model():
    global _model, _model_loaded_attempted
    if _model_loaded_attempted:
        return
    _model_loaded_attempted = True
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            _model = tf.keras.models.load_model(MODEL_PATH)
            log.info("keras_model_loaded", path=MODEL_PATH)
        except Exception as e:
            log.warning("keras_model_load_failed", error=str(e))
            _model = None
    else:
        log.info("keras_model_absent_using_rule_fallback", path=MODEL_PATH)


def classify(angles: dict, asana_hint: str = None) -> dict:
    """Returns {asana, score, corrections, source}. Uses Keras model when
    available to pick/confirm the asana + confidence; correctness detail
    (corrections) always comes from the transparent, auditable rule engine.
    """
    _try_load_model()

    if _model is not None:
        probs = _model.predict(_angles_to_vector(angles), verbose=0)[0]
        idx = int(np.argmax(probs))
        asana = ASANAS[idx]
        confidence = float(probs[idx])
        rule_result = check_pose(asana, angles)
        rule_result["confidence"] = round(confidence, 2)
        rule_result["source"] = "keras+rules"
        return rule_result

    # fallback: no trained model yet -> rule engine decides asana by best score
    asana = asana_hint or _best_matching_asana(angles)
    result = check_pose(asana, angles)
    result["confidence"] = result["score"]
    result["source"] = "rules_only"
    return result


def _best_matching_asana(angles: dict) -> str:
    best, best_score = ASANAS[0], -1.0
    for asana in ASANAS:
        s = check_pose(asana, angles)["score"]
        if s > best_score:
            best, best_score = asana, s
    return best
