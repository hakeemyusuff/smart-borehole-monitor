from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib

from app.ml.features import FEATURE_ORDER

logger = logging.getLogger(__name__)

# Module-level singletons, populated once at startup by load_model()
# The inference job and any route handler import these directly.

_model = None
_feature_columns: Optional[list[str]] = None

MODEL_PATH = Path("models/rf_level.joblib")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")


def load_model() -> None:
    """Load the trained model and and its feature-column order into module globals.
    Called once at startup. On any failure, leaves globals as None and logs
    """

    global _model, _feature_columns

    if not MODEL_PATH.exists():
        logger.warning(
            f"No model at {MODEL_PATH} - inference disabled until one is trained."
        )
        return

    if not FEATURE_COLUMNS_PATH.exists():
        logger.warning(
            f"Model present but {FEATURE_COLUMNS_PATH} - inference disabled "
            "(cannot verify feature order).",
        )

    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURE_COLUMNS_PATH) as f:
            columns = json.load(f)
    except Exception:
        logger.exception(
            "Failed to load model or feature columns - inference disabled."
        )
        return

    # checks to see if the model column fit the features column
    if columns != FEATURE_ORDER:
        logger.error(
            "Feature-column mismatch - model disabled. "
            f"On disk: {columns}. Expected: {FEATURE_ORDER}"
        )
        return
    
    _model = model
    _feature_columns = columns
    logger.info(
        f"Model loaded from {MODEL_PATH} ({len(columns)} features)."
    )
    
    
def get_model():
    """Return the loaded model or None if unavailable"""
    return _model

def get_features_columns() -> Optional[list[str]]:
    """Return the feature-column order the model was fit on, or None."""
    return _feature_columns