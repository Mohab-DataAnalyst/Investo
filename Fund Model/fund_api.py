import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from functools import lru_cache
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List, Union

# ── Artifact paths ────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# No separate MLB file — the MultiLabelBinarizerTransformer is embedded
# inside the saved pipeline and is loaded automatically with the model.
MODEL_FILENAME    = "funding_model.pkl"
METADATA_FILENAME = "funding_metadata.pkl"

router = APIRouter(prefix="/fund", tags=["fund"])


def _resolve_artifact_path(filename: str) -> Path:
    candidates = [BASE_DIR / filename, PROJECT_DIR / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{filename} not found. Searched: {', '.join(str(p) for p in candidates)}"
    )


@lru_cache(maxsize=1)
def _load_artifacts():
    # Only 2 artifacts — model (contains MLB inside pipeline) + metadata.
    return (
        joblib.load(_resolve_artifact_path(MODEL_FILENAME)),
        joblib.load(_resolve_artifact_path(METADATA_FILENAME)),
    )


# ── Request Schema ────────────────────────────────────────────────────
class ProjectInput(BaseModel):
    founded_year:           int
    country:                str
    industry:               str
    funding_stage:          str
    no_employees:           float
    social_media_followers: float
    community_id:           Union[str, int]
    degree_centrality:      float
    betweenness_centrality: float
    tech_stack:             Union[List[str], str]

    # Normalise to comma-separated string so the pipeline's internal
    # MultiLabelBinarizerTransformer receives what it was trained on.
    @field_validator("tech_stack", mode="before")
    @classmethod
    def normalize_tech_stack(cls, value):
        if isinstance(value, list):
            return ", ".join(str(i).strip().lower() for i in value if str(i).strip())
        if isinstance(value, str):
            return value.strip().lower()
        raise ValueError("tech_stack must be a list or comma-separated string")



# Category normalization — maps common casing/spelling variations to the
# exact strings the OneHotEncoder was fitted on during training.
_INDUSTRY_MAP = {
    "e-commerce": "E-commerce", "ecommerce": "E-commerce",
    "education & training": "Education & Training", "education": "Education & Training",
    "fintech": "FinTech", "fin tech": "FinTech",
    "food & beverage": "Food & Beverage", "food and beverage": "Food & Beverage",
    "healthcare clinics": "Healthcare Clinics", "healthcare": "Healthcare Clinics",
    "it services": "IT Services", "it": "IT Services",
    "logistics & delivery": "Logistics & Delivery", "logistics": "Logistics & Delivery",
    "marketing agencies": "Marketing Agencies", "marketing": "Marketing Agencies",
    "retail": "Retail",
    "saas": "SaaS",
}
_STAGE_MAP = {
    "seed": "Seed",
    "series a": "Series A", "series_a": "Series A",
    "series b": "Series B", "series_b": "Series B",
    "series c": "Series C", "series_c": "Series C",
    "ipo": "IPO",
}
_COUNTRY_MAP = {
    "australia": "Australia", "brazil": "Brazil", "canada": "Canada",
    "china": "China", "france": "France", "germany": "Germany",
    "india": "India", "japan": "Japan", "uk": "UK",
    "united kingdom": "UK", "usa": "USA", "united states": "USA",
    "us": "USA", "egypt": "Egypt",
}

def _norm(value: str, mapping: dict) -> str:
    return mapping.get(value.strip().lower(), value.strip())


# ── Preprocessing ─────────────────────────────────────────────────────
def preprocess_input(data: ProjectInput, metadata: dict) -> pd.DataFrame:
    df = pd.DataFrame([data.model_dump()])

    df["industry"]      = df["industry"].apply(lambda v: _norm(v, _INDUSTRY_MAP))
    df["country"]       = df["country"].apply(lambda v: _norm(v, _COUNTRY_MAP))
    df["funding_stage"] = df["funding_stage"].apply(lambda v: _norm(v, _STAGE_MAP))

    # Feature engineering — must match training exactly.
    df["startup_age"]  = datetime.now().year - df["founded_year"]
    df = df.drop(columns=["founded_year"])
    df["community_id"] = df["community_id"].astype(str)

    # tech_stack stays as comma-separated string — pipeline handles it.

    # Align column order to training feature order.
    feature_columns = metadata.get("feature_columns", [])
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

    return df


# ── Prediction Endpoint ───────────────────────────────────────────────
@router.post("/predict")
def predict(data: ProjectInput):
    try:
        model, metadata = _load_artifacts()
        processed_df    = preprocess_input(data, metadata)

        prediction = model.predict(processed_df)[0]

        # Floor at 0.1 to prevent negative predictions; no upper cap.
        prediction_rounded = float(max(round(prediction, 2), 0.1))

        return {
            "funding_prediction_raw":     float(prediction),
            "funding_prediction_rounded": prediction_rounded,
            "status": "success",
        }

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/")
def health():
    try:
        _load_artifacts()
        model_loaded = True
    except Exception:
        model_loaded = False
    return {"status": "Funding Prediction API is running", "model_loaded": model_loaded}


def create_app() -> FastAPI:
    app = FastAPI(title="Funding Prediction API")
    app.include_router(router)
    return app


app = create_app()
