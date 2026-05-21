"""
FastAPI приложение. Все эндпоинты NHL Salary Predictor.
"""

import io
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    SkaterInput, GoalieInput,
    PredictionOutput, PlayerSearchResult,
    BatchPredictionRow, BatchPredictionOutput,
)
from api.predictor import predictor, get_salary_tier

app = FastAPI(
    title="NHL Salary Predictor",
    description="Предсказание cap hit хоккеистов NHL по игровой статистике",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "skaters": {
            "model": "LightGBM + SelectFromModel",
            "n_features": len(predictor.skater_selected_cols),
            "test_mae": 1_247_000,
            "test_mape": 31.5,
            "test_r2": 0.667,
        },
        "goalies": {
            "model": "CatBoost",
            "test_mae": 1_251_000,
            "test_mape": 43.7,
            "test_r2": 0.529,
        },
        "seasons": "2021-22 → 2025-26",
        "target": "cap_hit (USD)",
    }


@app.get("/players/search", response_model=list[PlayerSearchResult])
def search_players(
        name: str = Query(..., min_length=2, description="Имя или фамилия игрока"),
        type: Literal["skater", "goalie"] = Query("skater"),
):
    results = predictor.search_players(name, type)
    if not results:
        raise HTTPException(status_code=404, detail=f"Игроки по запросу '{name}' не найдены")
    return results


@app.post("/predict/skater", response_model=PredictionOutput)
def predict_skater(inp: SkaterInput):
    try:
        return predictor.predict_skater(inp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/goalie", response_model=PredictionOutput)
def predict_goalie(inp: GoalieInput):
    try:
        return predictor.predict_goalie(inp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Только CSV файлы")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать CSV")

    required = {"gamesPlayed", "goals", "assists", "points", "timeOnIcePerGame"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Отсутствуют обязательные колонки: {missing}",
        )

    player_type = "goalie" if "savePct" in df.columns else "skater"

    predictions = []
    tier_dist: dict[str, int] = {}

    for _, row in df.iterrows():
        raw = row.to_dict()
        name = str(raw.get("skaterFullName") or raw.get("goalieFullName") or raw.get("name", "Unknown"))

        try:
            if player_type == "skater":
                inp = SkaterInput(**{k: v for k, v in raw.items()
                                     if k in SkaterInput.model_fields and not pd.isna(v)})
                result = predictor.predict_skater(inp)
            else:
                inp = GoalieInput(**{k: v for k, v in raw.items()
                                     if k in GoalieInput.model_fields and not pd.isna(v)})
                result = predictor.predict_goalie(inp)

            predictions.append(BatchPredictionRow(
                name=name,
                predicted_cap_hit=result.predicted_cap_hit,
                predicted_cap_hit_m=result.predicted_cap_hit_m,
                salary_tier=result.salary_tier,
            ))
            tier_dist[result.salary_tier] = tier_dist.get(result.salary_tier, 0) + 1

        except Exception:
            predictions.append(BatchPredictionRow(
                name=name,
                predicted_cap_hit=0.0,
                predicted_cap_hit_m="Ошибка",
                salary_tier="Unknown",
            ))

    return BatchPredictionOutput(
        count=len(predictions),
        predictions=predictions,
        tier_distribution=tier_dist,
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")