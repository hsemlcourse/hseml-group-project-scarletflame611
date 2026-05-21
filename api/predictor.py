"""
Загрузка моделей, препроцессинг, предсказание, SHAP, похожие игроки.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.append(str(Path(__file__).parent.parent))

from src.modeling import SelectThenPredict
from src.preprocessing import engineer_one_skater, engineer_one_goalie
from api.schemas import (
    SkaterInput, GoalieInput, PredictionOutput,
    ShapFeature, SimilarPlayer, PlayerSearchResult,
)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
FEATURES_DIR = ROOT / "data" / "features"

SALARY_TIERS = [
    (1_000_000,  "Entry-level",  "<$1M"),
    (3_000_000,  "Bottom-6",     "$1–3M"),
    (5_000_000,  "Middle-6",     "$3–5M"),
    (8_000_000,  "Top-6",        "$5–8M"),
    (12_000_000, "Star",         "$8–12M"),
    (float("inf"), "Franchise",  "$12M+"),
]

SKATER_KEY_STATS = {
    "C": ("points", "Очки"),
    "L": ("points", "Очки"),
    "R": ("points", "Очки"),
    "D": ("plusMinus", "+/-"),
}


def get_salary_tier(cap_hit: float) -> tuple[str, str]:
    for threshold, tier, tier_range in SALARY_TIERS:
        if cap_hit < threshold:
            return tier, tier_range
    return "Franchise", "$12M+"


class NHLPredictor:
    def __init__(self):
        self._load_models()
        self._load_datasets()

    def _load_models(self):
        skater_model = joblib.load(MODELS_DIR / "skaters_final.joblib")
        self.skater_selector = skater_model.selector
        self.skater_inner = skater_model.model
        self.skater_lgbm = self.skater_inner.named_steps["model"]
        self.skater_scaler = self.skater_inner.named_steps["scaler"]
        self.skater_selected_cols = skater_model.selected_cols
        self.skater_explainer = shap.TreeExplainer(self.skater_lgbm)

        goalie_pipeline = joblib.load(MODELS_DIR / "goalies_A_catboost.joblib")
        self.goalie_model = goalie_pipeline.named_steps["model"]
        self.goalie_scaler = goalie_pipeline.named_steps["scaler"]
        self.goalie_explainer = shap.TreeExplainer(self.goalie_model)

    def _load_datasets(self):
        sk = pd.read_csv(FEATURES_DIR / "skaters_test.csv")
        go = pd.read_csv(FEATURES_DIR / "goalies_test.csv")

        self.skaters_lookup = sk[
            ["playerId", "skaterFullName", "positionCode",
             "teamAbbrevs", "season_year", "cap_hit",
             "goals", "assists", "points", "plusMinus",
             "timeOnIcePerGame", "gamesPlayed"]
        ].dropna(subset=["cap_hit"]).copy()

        self.goalies_lookup = go[
            ["playerId", "goalieFullName", "teamAbbrevs",
             "season_year", "cap_hit", "wins",
             "savePct", "goalsAgainstAverage", "gamesStarted"]
        ].dropna(subset=["cap_hit"]).copy()

        # все числовые колонки для подстановки в форму при поиске
        self.skaters_full = sk.copy()
        self.goalies_full = go.copy()

    def _get_feature_cols(self, df: pd.DataFrame, exclude: list[str]) -> list[str]:
        return [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude]

    def _build_shap_top5(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        feature_values: np.ndarray,
    ) -> list[ShapFeature]:
        pairs = sorted(
            zip(feature_names, shap_values, feature_values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
        return [
            ShapFeature(
                feature=name,
                value=float(val),
                shap_value=float(sv),
                direction="up" if sv > 0 else "down",
            )
            for name, sv, val in pairs
        ]

    def _find_similar_skaters(
        self, predicted: float, position: str, n: int = 5
    ) -> list[SimilarPlayer]:
        low, high = predicted * 0.8, predicted * 1.2
        candidates = self.skaters_lookup[
            (self.skaters_lookup["cap_hit"] >= low) &
            (self.skaters_lookup["cap_hit"] <= high)
        ].copy()

        if candidates.empty:
            low, high = predicted * 0.7, predicted * 1.3
            candidates = self.skaters_lookup[
                (self.skaters_lookup["cap_hit"] >= low) &
                (self.skaters_lookup["cap_hit"] <= high)
            ].copy()

        candidates["dist"] = (candidates["cap_hit"] - predicted).abs()
        candidates = candidates.nsmallest(n, "dist")

        stat_col, stat_label = SKATER_KEY_STATS.get(position, ("points", "Очки"))
        result = []
        for _, row in candidates.iterrows():
            result.append(SimilarPlayer(
                name=row["skaterFullName"],
                position=row["positionCode"],
                team=str(row["teamAbbrevs"]),
                cap_hit=float(row["cap_hit"]),
                cap_hit_m=f"${row['cap_hit']/1e6:.2f}M",
                key_stat=stat_label,
                key_stat_value=float(row.get(stat_col, 0)),
            ))
        return result

    def _find_similar_goalies(
        self, predicted: float, n: int = 5
    ) -> list[SimilarPlayer]:
        low, high = predicted * 0.8, predicted * 1.2
        candidates = self.goalies_lookup[
            (self.goalies_lookup["cap_hit"] >= low) &
            (self.goalies_lookup["cap_hit"] <= high)
        ].copy()

        if candidates.empty:
            low, high = predicted * 0.7, predicted * 1.3
            candidates = self.goalies_lookup[
                (self.goalies_lookup["cap_hit"] >= low) &
                (self.goalies_lookup["cap_hit"] <= high)
            ].copy()

        candidates["dist"] = (candidates["cap_hit"] - predicted).abs()
        candidates = candidates.nsmallest(n, "dist")

        result = []
        for _, row in candidates.iterrows():
            result.append(SimilarPlayer(
                name=row["goalieFullName"],
                position="G",
                team=str(row["teamAbbrevs"]),
                cap_hit=float(row["cap_hit"]),
                cap_hit_m=f"${row['cap_hit']/1e6:.2f}M",
                key_stat="SV%",
                key_stat_value=float(row.get("savePct", 0)),
            ))
        return result

    def predict_skater(self, inp: SkaterInput) -> PredictionOutput:
        raw = inp.model_dump(exclude={"prev_season"})
        prev = inp.prev_season.model_dump() if inp.prev_season else None

        df = engineer_one_skater(raw, prev_season=prev)

        x_all = self.skaters_full.iloc[:1]  # берём структуру колонок
        exclude = [
            "playerId", "skaterFullName", "goalieFullName", "lastName",
            "season", "seasonId", "season_year", "teamAbbrevs",
            "positionCode", "shootsCatches", "birthCity", "birthCountry",
            "name_key", "cap_hit", "log_cap_hit",
            "player_name_cw", "team_cw", "position_cw",
            "evPoints", "ppTimeOnIce", "ppTimeOnIcePctPerGame",
            "missedShotWideOfNet", "ppIndividualSatFor",
            "timeOnIce", "completeGames", "gamesPlayed",
            "saves", "regulationWins", "country_group",
            "cap_pct", "cap_pct_lag1", "salary_cap",
            "cap_hit_lag1", "cap_hit_delta",
        ]
        feature_cols = self.skater_selected_cols

        # выравниваем колонки: добавляем отсутствующие с нулями
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        x = df[feature_cols]

        x_sel = pd.DataFrame(
            self.skater_selector.transform(x),
            columns=self.skater_selected_cols,
        )
        x_scaled = pd.DataFrame(
            self.skater_scaler.transform(x_sel),
            columns=self.skater_selected_cols,
        )

        pred_log = self.skater_lgbm.predict(x_scaled)[0]
        predicted = float(np.expm1(pred_log))

        shap_vals = self.skater_explainer.shap_values(x_scaled)[0]
        shap_top5 = self._build_shap_top5(
            shap_vals, self.skater_selected_cols, x_scaled.values[0]
        )

        tier, tier_range = get_salary_tier(predicted)
        similar = self._find_similar_skaters(predicted, inp.positionCode)

        return PredictionOutput(
            predicted_cap_hit=predicted,
            predicted_cap_hit_m=f"${predicted/1e6:.2f}M",
            salary_tier=tier,
            tier_range=tier_range,
            shap_top5=shap_top5,
            similar_players=similar,
        )

    def predict_goalie(self, inp: GoalieInput) -> PredictionOutput:
        raw = inp.model_dump(exclude={"prev_season"})
        prev = inp.prev_season.model_dump() if inp.prev_season else None

        df = engineer_one_goalie(raw, prev_season=prev)

        feature_cols = [c for c in self.goalies_full.select_dtypes(
            include=[np.number]).columns
            if c not in [
                "playerId", "season", "seasonId", "season_year",
                "cap_hit", "log_cap_hit", "timeOnIce", "completeGames",
                "gamesPlayed", "saves", "regulationWins",
                "cap_hit_lag1", "cap_hit_delta", "country_group",
            ]
        ]

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        x = df[feature_cols]

        x_scaled = pd.DataFrame(
            self.goalie_scaler.transform(x),
            columns=feature_cols,
        )

        pred_log = self.goalie_model.predict(x_scaled)[0]
        predicted = float(np.expm1(pred_log))

        shap_vals = self.goalie_explainer.shap_values(x_scaled)[0]
        shap_top5 = self._build_shap_top5(
            shap_vals, feature_cols, x_scaled.values[0]
        )

        tier, tier_range = get_salary_tier(predicted)
        similar = self._find_similar_goalies(predicted)

        return PredictionOutput(
            predicted_cap_hit=predicted,
            predicted_cap_hit_m=f"${predicted/1e6:.2f}M",
            salary_tier=tier,
            tier_range=tier_range,
            shap_top5=shap_top5,
            similar_players=similar,
        )

    def search_players(self, name: str, player_type: str) -> list[PlayerSearchResult]:
        name_lower = name.lower().strip()

        if player_type == "skater":
            df = self.skaters_lookup
            name_col = "skaterFullName"
            pos_col = "positionCode"
        else:
            df = self.goalies_lookup
            name_col = "goalieFullName"
            pos_col = None

        mask = df[name_col].str.lower().str.contains(name_lower, na=False)
        matches = df[mask].head(10)

        results = []
        for _, row in matches.iterrows():
            stats = {
                "gamesPlayed": int(row.get("gamesPlayed", 0)),
            }
            if player_type == "skater":
                stats.update({
                    "goals": int(row.get("goals", 0)),
                    "assists": int(row.get("assists", 0)),
                    "points": int(row.get("points", 0)),
                    "plusMinus": int(row.get("plusMinus", 0)),
                    "timeOnIcePerGame": float(row.get("timeOnIcePerGame", 0)),
                })
            else:
                stats.update({
                    "wins": int(row.get("wins", 0)),
                    "savePct": float(row.get("savePct", 0)),
                    "goalsAgainstAverage": float(row.get("goalsAgainstAverage", 0)),
                    "gamesStarted": int(row.get("gamesStarted", 0)),
                })

            results.append(PlayerSearchResult(
                player_id=int(row.get("playerId", 0)),
                name=row[name_col],
                position=str(row.get(pos_col, "G")) if pos_col else "G",
                team=str(row.get("teamAbbrevs", "")),
                season_year=int(row.get("season_year", 2026)),
                cap_hit=float(row["cap_hit"]),
                cap_hit_m=f"${row['cap_hit']/1e6:.2f}M",
                stats=stats,
            ))
        return results


predictor = NHLPredictor()