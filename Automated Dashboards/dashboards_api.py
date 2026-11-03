from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from DataAccess.db import load_interactions, load_investors, load_startups
from .dashboards import (
    get_investor_dashboard,
    get_market_dashboard,
    get_risk_dashboard,
    get_startup_dashboard,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


class StartupInput(BaseModel):
    startup_name: str


class InvestorInput(BaseModel):
    investor_id: str


def _normalize_text(series, default="unknown"):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace("", default)
    )


def _fill_numeric(df, column, default):
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)


def _prepare_startups(df):
    df = df.copy()
    df["startup_name"] = df["startup_name"].fillna("").astype(str)
    for col in ["industry", "country", "funding_stage"]:
        df[col] = _normalize_text(df[col])

    for col in [
        "annual_revenue",
        "total_funding",
        "no_employees",
        "valuation",
        "customer_base",
        "social_media_followers",
        "degree_centrality",
        "betweenness_centrality",
    ]:
        _fill_numeric(df, col, 0)
    _fill_numeric(df, "success_score", 5)

    _fill_numeric(df, "founded_year", pd.Timestamp.now().year)
    df["community_id"] = (
        pd.to_numeric(df["community_id"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    df["ipo"] = df["ipo"].fillna(False).astype(bool)
    df["acquired"] = df["acquired"].fillna(False).astype(bool)
    return df


def _prepare_investors(df):
    df = df.copy()
    df["investor_id"] = df["investor_id"].fillna("").astype(str)
    for col in ["preferred_industry", "country_preference", "risk_tolerance"]:
        df[col] = _normalize_text(df[col])
    return df


def _prepare_interactions(df):
    df = df.copy()
    df["investor_id"] = df["investor_id"].fillna("").astype(str)
    df["startup_name"] = df["startup_name"].fillna("").astype(str)
    return df


@lru_cache(maxsize=1)
def _load_dashboard_data():
    return (
        _prepare_startups(load_startups()),
        _prepare_investors(load_investors()),
        _prepare_interactions(load_interactions()),
    )


def _data_or_503():
    try:
        return _load_dashboard_data()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Dashboard data unavailable: {e}",
        ) from e


def _return_or_404(result):
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/startup")
def startup_dashboard(data: StartupInput):
    startups_df, _, _ = _data_or_503()
    try:
        return _return_or_404(get_startup_dashboard(data.startup_name, startups_df))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/investor")
def investor_dashboard(data: InvestorInput):
    startups_df, investors_df, interactions_df = _data_or_503()
    try:
        return _return_or_404(
            get_investor_dashboard(
                data.investor_id,
                investors_df,
                startups_df,
                interactions_df,
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/market")
def market_dashboard():
    startups_df, _, _ = _data_or_503()
    try:
        return get_market_dashboard(startups_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/risk")
def risk_dashboard():
    startups_df, _, _ = _data_or_503()
    try:
        return get_risk_dashboard(startups_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/")
def health():
    try:
        startups_df, investors_df, _ = _load_dashboard_data()
        return {
            "status": "Dashboards API is running",
            "data_loaded": True,
            "startups": len(startups_df),
            "investors": len(investors_df),
        }
    except Exception as e:
        return {
            "status": "Dashboards API is unavailable",
            "data_loaded": False,
            "detail": str(e),
        }


def create_app() -> FastAPI:
    app = FastAPI(title="Startup Dashboards API")
    app.include_router(router)
    return app


app = create_app()
