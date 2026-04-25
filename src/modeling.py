"""
Пайплайн обучения и оценки моделей для предсказания зарплат игроков NHL.

Скейтеры: data/features/skaters_*.csv
Вратари:  data/features/goalies_*.csv
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn import set_config
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

set_config(transform_output="pandas")

random_state = 42
features_dir = Path("../data/features")
models_dir = Path("../models")
target_col = "log_cap_hit"

# колонки которые никогда не идут в модель
exclude_cols = [
    "playerId", "skaterFullName", "goalieFullName", "lastName",
    "season", "seasonId", "season_year", "teamAbbrevs",
    "positionCode", "shootsCatches", "birthCity", "birthCountry",
    "name_key", "cap_hit", "log_cap_hit",
    "player_name_cw", "team_cw", "position_cw",
    # мультиколлинеарные дубли выявленные в EDA
    "evPoints",           # r=0.96 с points
    "ppTimeOnIce",        # r=0.99 с ppTimeOnIcePerGame
    "ppTimeOnIcePctPerGame",  # r=0.99 с ppTimeOnIcePerGame
    "missedShotWideOfNet",    # r=0.99 с missedShots
    "ppIndividualSatFor",     # r=0.99 с ppShots
    # вратари — дубли объёма
    "timeOnIce",          # r=1.00 с gamesStarted
    "completeGames",      # r=0.99 с gamesStarted
    "gamesPlayed",        # r=0.99 с gamesStarted
    "saves",              # r=0.999 с shotsAgainst
    "regulationWins",     # r=0.99 с wins
]


# --- утилиты ---

def setup_dirs():
    models_dir.mkdir(parents=True, exist_ok=True)


def load_splits(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает train/val/test сплиты для скейтеров или вратарей."""
    train = pd.read_csv(features_dir / f"{prefix}_train.csv")
    val = pd.read_csv(features_dir / f"{prefix}_val.csv")
    test = pd.read_csv(features_dir / f"{prefix}_test.csv")
    log.info("%s: train=%d, val=%d, test=%d", prefix, len(train), len(val), len(test))
    return train, val, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Возвращает список признаков: числовые колонки за вычетом exclude_cols."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric if c not in exclude_cols]
    return features


def evaluate(
    model,
    x: pd.DataFrame,
    y: pd.Series,
    label: str = "",
) -> dict:
    """
    Считает метрики в исходном пространстве (USD) через expm1.
    Таргет — log_cap_hit, предсказания переводим обратно.
    """
    pred_log = model.predict(x)
    pred = np.expm1(pred_log)
    actual = np.expm1(y.values)

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100

    if label:
        log.info("%s — MAE: %.0f, RMSE: %.0f, R2: %.3f, MAPE: %.1f%%",
                 label, mae, rmse, r2, mape)

    return {"label": label, "mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def log_experiment(results: list[dict], row: dict):
    """Добавляет строку в таблицу экспериментов."""
    results.append(row)


def save_model(pipeline, name: str):
    path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, path)
    log.info("Модель сохранена: %s", path)


def load_model(name: str):
    path = models_dir / f"{name}.joblib"
    return joblib.load(path)


def build_pipeline(model) -> Pipeline:
    """StandardScaler + модель. Скейлер обучается только на train."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def print_results_table(results: list[dict]):
    df = pd.DataFrame(results)
    df["mae"] = df["mae"].apply(lambda x: f"{x/1e6:.3f}M")
    df["rmse"] = df["rmse"].apply(lambda x: f"{x/1e6:.3f}M")
    df["r2"] = df["r2"].round(3)
    df["mape"] = df["mape"].round(1).astype(str) + "%"
    print(df[["label", "mae", "rmse", "r2", "mape"]].to_string(index=False))


# --- baseline ---

def run_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    results: list[dict],
    prefix: str,
):
    """
    LinearRegression без скейлинга и без feature engineering —
    нижняя граница качества, точка отсчёта для всех остальных моделей.
    """
    log.info("Baseline: LinearRegression")
    model = LinearRegression()
    model.fit(x_train, y_train)
    row = evaluate(model, x_val, y_val, label=f"{prefix}_baseline_linear")
    log_experiment(results, row)
    save_model(model, f"{prefix}_baseline")
    return model


# --- основные модели ---

def run_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    results: list[dict],
    prefix: str,
) -> dict:
    """
    Ridge, RandomForest, XGBoost, LightGBM — с Pipeline (StandardScaler + модель).
    Параметры по умолчанию на первом запуске, гиперпараметры — на cp2.
    """
    models = {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        ),
    }

    trained = {}
    for name, estimator in models.items():
        log.info("Обучаем: %s", name)
        pipeline = build_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        row = evaluate(pipeline, x_val, y_val, label=f"{prefix}_{name}")
        log_experiment(results, row)
        save_model(pipeline, f"{prefix}_{name}")
        trained[name] = pipeline

    return trained


# --- уменьшение размерности ---

def run_dim_reduction(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    results: list[dict],
    prefix: str,
):
    """
    Два эксперимента с уменьшением размерности:
    1. SelectFromModel на основе LightGBM — отбор признаков по важности
    2. PCA до n компонент объясняющих 95% дисперсии + Ridge
    Сравниваем с полным набором признаков.
    """
    log.info("Уменьшение размерности: SelectFromModel")
    selector_base = LGBMRegressor(
        n_estimators=200, random_state=random_state, verbose=-1
    )
    selector = SelectFromModel(selector_base, threshold="median")
    selector.fit(x_train, y_train)

    selected_cols = x_train.columns[selector.get_support()].tolist()
    x_train_sel = pd.DataFrame(selector.transform(x_train), columns=selected_cols)
    x_val_sel = pd.DataFrame(selector.transform(x_val), columns=selected_cols)
    n_selected = len(selected_cols)
    log.info("SelectFromModel: отобрано %d из %d признаков", n_selected, x_train.shape[1])

    model_sel = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LGBMRegressor(
            n_estimators=300, learning_rate=0.05,
            random_state=random_state, verbose=-1
        )),
    ])
    model_sel.fit(x_train_sel, y_train)
    row = evaluate(model_sel, x_val_sel, y_val,
                   label=f"{prefix}_lgbm_select_{n_selected}feat")
    log_experiment(results, row)

    log.info("Уменьшение размерности: PCA + Ridge")
    pca_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=random_state)),
        ("model", Ridge(alpha=1.0)),
    ])
    pca_pipeline.fit(x_train, y_train)
    n_components = pca_pipeline.named_steps["pca"].n_components_
    log.info("PCA: %d компонент объясняют 95%% дисперсии", n_components)

    row = evaluate(pca_pipeline, x_val, y_val,
                   label=f"{prefix}_pca_{n_components}comp_ridge")
    log_experiment(results, row)

    return selector, pca_pipeline


# --- финальная оценка на test ---

def evaluate_on_test(
    model_name: str,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    prefix: str,
):
    """
    Вызывается только один раз после выбора финальной модели.
    Оценивает на test-сете который не участвовал в обучении и валидации.
    """
    log.info("Финальная оценка на test: %s", model_name)
    model = load_model(f"{prefix}_{model_name}")
    metrics = evaluate(model, x_test, y_test, label=f"{prefix}_{model_name}_TEST")
    return metrics


# --- точки входа ---

def run_skaters():
    setup_dirs()
    results = []

    train, val, test = load_splits("skaters")
    feature_cols = get_feature_cols(train)
    log.info("Признаков скейтеры: %d", len(feature_cols))

    x_train = train[feature_cols]
    y_train = train[target_col]
    x_val = val[feature_cols]
    y_val = val[target_col]
    x_test = test[feature_cols]
    y_test = test[target_col]

    run_baseline(x_train, y_train, x_val, y_val, results, prefix="skaters")
    run_models(x_train, y_train, x_val, y_val, results, prefix="skaters")
    run_dim_reduction(x_train, y_train, x_val, y_val, results, prefix="skaters")

    log.info("Таблица экспериментов скейтеры:")
    print_results_table(results)

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/skaters_experiments.csv", index=False)

    return results, x_test, y_test


def run_goalies():
    setup_dirs()
    results = []

    train, val, test = load_splits("goalies")
    feature_cols = get_feature_cols(train)
    log.info("Признаков вратари: %d", len(feature_cols))

    x_train = train[feature_cols]
    y_train = train[target_col]
    x_val = val[feature_cols]
    y_val = val[target_col]
    x_test = test[feature_cols]
    y_test = test[target_col]

    run_baseline(x_train, y_train, x_val, y_val, results, prefix="goalies")
    run_models(x_train, y_train, x_val, y_val, results, prefix="goalies")
    run_dim_reduction(x_train, y_train, x_val, y_val, results, prefix="goalies")

    log.info("Таблица экспериментов вратари:")
    print_results_table(results)

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/goalies_experiments.csv", index=False)

    return results, x_test, y_test


def run():
    run_skaters()
    run_goalies()


if __name__ == "__main__":
    run()
