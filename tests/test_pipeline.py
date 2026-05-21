"""
Smoke-тесты пайплайна NHL Salary Prediction.
Проверяют что данные загружаются корректно, preprocessing не ломает структуру,
сплиты не пересекаются и модель предсказывает.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from modeling import SelectThenPredict  # noqa: F401

features_dir = Path("data/features")
models_dir = Path("models")
raw_dir = Path("data/raw")


# --- фикстуры ---


@pytest.fixture(scope="module")
def skaters_train():
    return pd.read_csv(features_dir / "skaters_train.csv")


@pytest.fixture(scope="module")
def skaters_val():
    return pd.read_csv(features_dir / "skaters_val.csv")


@pytest.fixture(scope="module")
def skaters_test():
    return pd.read_csv(features_dir / "skaters_test.csv")


@pytest.fixture(scope="module")
def goalies_train():
    return pd.read_csv(features_dir / "goalies_train.csv")


@pytest.fixture(scope="module")
def final_model():
    return joblib.load(models_dir / "skaters_final.joblib")


# --- тесты данных ---


def test_skaters_train_not_empty(skaters_train):
    assert len(skaters_train) > 0, "Train скейтеров пустой"


def test_skaters_required_columns(skaters_train):
    required = ["playerId", "season_year", "cap_hit", "log_cap_hit", "gamesPlayed"]
    for col in required:
        assert col in skaters_train.columns, f"Колонка {col} отсутствует в train"


def test_log_target_invertible(skaters_train):
    """log1p / expm1 корректно инвертируется."""
    original = skaters_train["cap_hit"].values
    reconstructed = np.expm1(skaters_train["log_cap_hit"].values)
    np.testing.assert_allclose(original, reconstructed, rtol=1e-5)


def test_no_nulls_in_target(skaters_train, skaters_val, skaters_test):
    for df, name in [(skaters_train, "train"), (skaters_val, "val"), (skaters_test, "test")]:
        nulls = df["log_cap_hit"].isna().sum()
        assert nulls == 0, f"NaN в log_cap_hit в {name}: {nulls} строк"


def test_cap_hit_range(skaters_train):
    """Все зарплаты в разумном диапазоне НХЛ."""
    assert skaters_train["cap_hit"].min() >= 500_000, "Зарплата ниже минимума НХЛ"
    assert skaters_train["cap_hit"].max() <= 25_000_000, "Зарплата выше разумного максимума"


# --- тесты сплита ---


def test_temporal_split_no_overlap(skaters_train, skaters_val, skaters_test):
    """Сезоны train/val/test не пересекаются."""
    train_seasons = set(skaters_train["season_year"].unique())
    val_seasons = set(skaters_val["season_year"].unique())
    test_seasons = set(skaters_test["season_year"].unique())

    assert train_seasons & val_seasons == set(), (
        f"Пересечение train и val: {train_seasons & val_seasons}"
    )
    assert val_seasons & test_seasons == set(), (
        f"Пересечение val и test: {val_seasons & test_seasons}"
    )
    assert train_seasons & test_seasons == set(), (
        f"Пересечение train и test: {train_seasons & test_seasons}"
    )


def test_temporal_split_order(skaters_train, skaters_val, skaters_test):
    """train < val < test по сезонам."""
    assert skaters_train["season_year"].max() < skaters_val["season_year"].min(), (
        "train содержит сезоны позже val"
    )
    assert skaters_val["season_year"].max() < skaters_test["season_year"].min(), (
        "val содержит сезоны позже test"
    )


def test_no_player_leakage(skaters_train, skaters_test):
    """Один игрок может быть в train и test (разные сезоны) — это нормально.
    Но одна и та же строка (игрок + сезон) не должна быть в обоих."""
    train_keys = set(zip(skaters_train["playerId"], skaters_train["season_year"]))
    test_keys = set(zip(skaters_test["playerId"], skaters_test["season_year"]))
    overlap = train_keys & test_keys
    assert len(overlap) == 0, f"Утечка данных: {len(overlap)} строк в train и test"


# --- тесты goalies ---


def test_goalies_train_not_empty(goalies_train):
    assert len(goalies_train) > 0, "Train вратарей пустой"


def test_goalies_required_columns(goalies_train):
    required = ["playerId", "season_year", "cap_hit", "log_cap_hit", "gamesStarted"]
    for col in required:
        assert col in goalies_train.columns, f"Колонка {col} отсутствует"


# --- тесты модели ---


def test_model_loads(final_model):
    assert final_model is not None, "Модель не загрузилась"


def test_model_predicts(final_model, skaters_test):
    exclude_cols = [
        "playerId",
        "skaterFullName",
        "goalieFullName",
        "lastName",
        "season",
        "seasonId",
        "season_year",
        "teamAbbrevs",
        "positionCode",
        "shootsCatches",
        "birthCity",
        "birthCountry",
        "name_key",
        "cap_hit",
        "log_cap_hit",
        "player_name_cw",
        "team_cw",
        "position_cw",
        "evPoints",
        "ppTimeOnIce",
        "ppTimeOnIcePctPerGame",
        "missedShotWideOfNet",
        "ppIndividualSatFor",
        "timeOnIce",
        "completeGames",
        "gamesPlayed",
        "saves",
        "regulationWins",
        "birthCountry",
        "country_group",
        "cap_pct",
        "cap_pct_lag1",
        "salary_cap",
    ]
    numeric = skaters_test.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric if c not in exclude_cols]
    x_test = skaters_test[feature_cols]

    preds = final_model.predict(x_test)
    assert len(preds) == len(x_test), "Количество предсказаний не совпадает"
    assert not np.any(np.isnan(preds)), "NaN в предсказаниях"


def test_model_predictions_in_range(final_model, skaters_test):
    """Предсказания в разумном диапазоне после expm1."""
    exclude_cols = [
        "playerId",
        "skaterFullName",
        "goalieFullName",
        "lastName",
        "season",
        "seasonId",
        "season_year",
        "teamAbbrevs",
        "positionCode",
        "shootsCatches",
        "birthCity",
        "birthCountry",
        "name_key",
        "cap_hit",
        "log_cap_hit",
        "player_name_cw",
        "team_cw",
        "position_cw",
        "evPoints",
        "ppTimeOnIce",
        "ppTimeOnIcePctPerGame",
        "missedShotWideOfNet",
        "ppIndividualSatFor",
        "timeOnIce",
        "completeGames",
        "gamesPlayed",
        "saves",
        "regulationWins",
        "birthCountry",
        "country_group",
        "cap_pct",
        "cap_pct_lag1",
        "salary_cap",
    ]
    numeric = skaters_test.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric if c not in exclude_cols]
    x_test = skaters_test[feature_cols]

    preds_usd = np.expm1(final_model.predict(x_test))
    assert preds_usd.min() > 0, "Предсказания содержат отрицательные зарплаты"
    assert preds_usd.max() < 30_000_000, "Предсказания выше разумного максимума"
