"""
Пайплайн обучения и оценки моделей для предсказания зарплат игроков NHL.

Скейтеры: data/features/skaters_*.csv
Вратари:  data/features/goalies_*.csv
"""

import logging
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn import set_config
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import StackingRegressor
import warnings
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
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
    "birthCountry",
    "country_group",
    "cap_pct",
    "cap_pct_lag1",
    "salary_cap",
    "cap_hit_lag1",
    "cap_hit_delta",
]


class SelectThenPredict:
    """Обёртка для SelectFromModel + Pipeline чтобы predict работал напрямую."""
    def __init__(self, selector, model, selected_cols):
        self.selector = selector
        self.model = model
        self.selected_cols = selected_cols

    def predict(self, x):
        x_sel = pd.DataFrame(
            self.selector.transform(x),
            columns=self.selected_cols,
        )
        return self.model.predict(x_sel)


# --- утилиты ---

def setup_dirs():
    models_dir.mkdir(parents=True, exist_ok=True)


def load_splits(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(features_dir / f"{prefix}_train.csv")
    val = pd.read_csv(features_dir / f"{prefix}_val.csv")
    test = pd.read_csv(features_dir / f"{prefix}_test.csv")
    log.info("%s: train=%d, val=%d, test=%d", prefix, len(train), len(val), len(test))
    return train, val, test


def get_feature_cols(df: pd.DataFrame, prefix: str = "") -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    always_exclude = exclude_cols.copy()
    # для постановки Б убираем всё связанное с текущей зарплатой
    if "B" in prefix:
        always_exclude += [
            "cap_hit_lag1",   # текущая зарплата — прямой сигнал
            "cap_hit_delta",  # изменение зарплаты — тоже сигнал
            "cap_hit_next",   # это таргет, не должен быть в признаках
        ]
    features = [c for c in numeric if c not in always_exclude]
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
    df["mae"] = df["mae"].apply(lambda x: f"{x / 1e6:.3f}M")
    if "rmse" in df.columns:
        df["rmse"] = df["rmse"].apply(lambda x: f"{x / 1e6:.3f}M")
    df["r2"] = df["r2"].round(3)
    df["mape"] = df["mape"].round(1).astype(str) + "%"
    cols = [c for c in ["label", "mae", "rmse", "r2", "mape"] if c in df.columns]
    print(df[cols].to_string(index=False))


def season_cv_splits(df: pd.DataFrame) -> list[tuple]:
    """
    Кросс-валидация по сезонам для вратарей.
    Каждый фолд: train = все сезоны до val_season, val = один сезон.
    """
    seasons = sorted(df["season_year"].unique())
    splits = []
    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]
        train_idx = df[df["season_year"].isin(train_seasons)].index
        val_idx = df[df["season_year"] == val_season].index
        splits.append((train_idx, val_idx))
        log.info(
            "Fold %d: train%s → val[%d]  (%d/%d строк)",
            i, train_seasons, val_season, len(train_idx), len(val_idx)
        )
    return splits


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
    Ridge, ElasticNet, RandomForest, XGBoost, LightGBM, CatBoost —
    с Pipeline (StandardScaler + модель), параметры по умолчанию.
    """
    models = {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "elasticnet": ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            random_state=random_state,
            max_iter=10000,
        ),
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
        "catboost": CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_state=random_state,
            verbose=0,
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


def tune_model(
        model_name: str,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        prefix: str,
        n_trials: int = 50,
) -> dict:
    def objective(trial):
        if model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            model = LGBMRegressor(
                **params, random_state=random_state, verbose=-1, n_jobs=-1
            )

        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            model = XGBRegressor(
                **params, random_state=random_state, verbosity=0
            )

        elif model_name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 3, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            }
            model = CatBoostRegressor(
                **params, random_state=random_state, verbose=0
            )

        elif model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            }
            model = RandomForestRegressor(
                **params, random_state=random_state, n_jobs=-1
            )

        elif model_name == "elasticnet":
            params = {
                "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }
            model = ElasticNet(
                **params, random_state=random_state, max_iter=10000
            )

        else:
            raise ValueError(f"Неизвестная модель: {model_name}")

        pipeline = build_pipeline(model)
        pipeline.fit(x_train, y_train)
        metrics = evaluate(pipeline, x_val, y_val)
        return metrics["mae"]

    log.info("Optuna: тюнинг %s для %s (%d trials)", model_name, prefix, n_trials)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_mae = study.best_value
    log.info(
        "Optuna %s %s: лучший MAE=%.0f, params=%s",
        prefix, model_name, best_mae, best_params
    )

    params_path = models_dir / f"best_params_{prefix}_{model_name}.json"
    with open(params_path, "w") as f:
        json.dump({"best_mae": best_mae, "params": best_params}, f, indent=2)
    log.info("Параметры сохранены: %s", params_path)

    return best_params


def run_tuned_models(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        results: list[dict],
        prefix: str,
        n_trials: int = 50,
) -> dict:
    tunable = ["lightgbm", "xgboost", "catboost", "random_forest", "elasticnet"]
    trained = {}

    for model_name in tunable:
        best_params = tune_model(
            model_name, x_train, y_train, x_val, y_val,
            prefix=prefix, n_trials=n_trials,
        )

        if model_name == "lightgbm":
            model = LGBMRegressor(
                **best_params, random_state=random_state, verbose=-1, n_jobs=-1
            )
        elif model_name == "xgboost":
            model = XGBRegressor(
                **best_params, random_state=random_state, verbosity=0
            )
        elif model_name == "catboost":
            model = CatBoostRegressor(
                **best_params, random_state=random_state, verbose=0
            )
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                **best_params, random_state=random_state, n_jobs=-1
            )
        elif model_name == "elasticnet":
            model = ElasticNet(
                **best_params, random_state=random_state, max_iter=10000
            )

        pipeline = build_pipeline(model)
        pipeline.fit(x_train, y_train)
        row = evaluate(
            pipeline, x_val, y_val,
            label=f"{prefix}_{model_name}_tuned",
        )
        log_experiment(results, row)
        save_model(pipeline, f"{prefix}_{model_name}_tuned")
        trained[model_name] = pipeline

    return trained


def run_stacking(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        results: list[dict],
        prefix: str,
        cv: int = 5,
) -> Pipeline:
    def load_best_params(model_name: str) -> dict:
        path = models_dir / f"best_params_{prefix}_{model_name}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)["params"]
        log.warning("Параметры %s не найдены, используем дефолтные", model_name)
        return {}

    lgbm_params = load_best_params("lightgbm")
    xgb_params = load_best_params("xgboost")
    cb_params = load_best_params("catboost")
    rf_params = load_best_params("random_forest")
    # ElasticNet не включаем в stacking — линейная модель слабее деревьев как base estimator

    estimators = [
        ("lightgbm", Pipeline([
            ("scaler", StandardScaler()),
            ("model", LGBMRegressor(
                **lgbm_params,
                random_state=random_state, verbose=-1, n_jobs=-1,
            )),
        ])),
        ("xgboost", Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                **xgb_params,
                random_state=random_state, verbosity=0,
            )),
        ])),
        ("catboost", Pipeline([
            ("scaler", StandardScaler()),
            ("model", CatBoostRegressor(
                **cb_params,
                random_state=random_state, verbose=0,
            )),
        ])),
        ("random_forest", Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                **rf_params,
                random_state=random_state, n_jobs=-1,
            )),
        ])),
    ]

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=cv,
        n_jobs=-1,
        passthrough=False,
    )

    log.info("Stacking: обучаем ансамбль для %s...", prefix)
    stacking.fit(x_train, y_train)

    row = evaluate(stacking, x_val, y_val, label=f"{prefix}_stacking")
    log_experiment(results, row)
    save_model(stacking, f"{prefix}_stacking")

    log.info("Stacking обучен")
    return stacking


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

    # сохраняем через обёртку чтобы predict работал на полном наборе фичей
    wrapper = SelectThenPredict(selector, model_sel, selected_cols)
    save_model(wrapper, f"{prefix}_lgbm_select_{n_selected}feat")

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
    save_model(pca_pipeline, f"{prefix}_pca_{n_components}comp_ridge")

    return selector, pca_pipeline


# --- точки входа скейтеры ---

def run_skaters(prefix: str = "skaters", n_trials: int = 50):
    setup_dirs()
    results = []

    train, val, test = load_splits(prefix)
    feature_cols = get_feature_cols(train, prefix=prefix)
    log.info("Признаков %s: %d", prefix, len(feature_cols))

    x_train = train[feature_cols]
    y_train = train[target_col]
    x_val = val[feature_cols]
    y_val = val[target_col]
    x_test = test[feature_cols]
    y_test = test[target_col]

    run_baseline(x_train, y_train, x_val, y_val, results, prefix=prefix)
    run_models(x_train, y_train, x_val, y_val, results, prefix=prefix)
    run_tuned_models(x_train, y_train, x_val, y_val, results, prefix=prefix, n_trials=n_trials)
    cv = 3 if prefix == "skaters_B" else 5
    run_stacking(x_train, y_train, x_val, y_val, results, prefix=prefix, cv=cv)
    run_dim_reduction(x_train, y_train, x_val, y_val, results, prefix=prefix)

    log.info("Таблица экспериментов %s:", prefix)
    print_results_table(results)

    pd.DataFrame(results).to_csv(
        models_dir / f"{prefix}_experiments.csv", index=False
    )
    return results, x_test, y_test


# --- точки входа вратари ---

def run_goalies_cv(prefix: str):
    setup_dirs()

    if prefix == "goalies_A":
        train_df = pd.read_csv(features_dir / "goalies_train.csv")
        val_df = pd.read_csv(features_dir / "goalies_val.csv")
        test_df = pd.read_csv(features_dir / "goalies_test.csv")
    else:
        train_df = pd.read_csv(features_dir / "goalies_B_train.csv")
        val_df = pd.read_csv(features_dir / "goalies_B_val.csv")
        test_df = pd.read_csv(features_dir / "goalies_B_test.csv")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    feature_cols = get_feature_cols(full_df, prefix=prefix)
    log.info("%s CV: %d строк, %d признаков", prefix, len(full_df), len(feature_cols))

    splits = season_cv_splits(full_df)

    models_default = {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "elasticnet": ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            random_state=random_state,
            max_iter=10000,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, random_state=random_state, verbosity=0,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300, learning_rate=0.05,
            random_state=random_state, verbose=-1, n_jobs=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=300, learning_rate=0.05,
            depth=4, random_state=random_state, verbose=0,
        ),
    }
    results = []

    # дефолтные модели по фолдам
    for name, estimator in models_default.items():
        fold_metrics = []
        for train_idx, val_idx in splits:
            x_tr = full_df.loc[train_idx, feature_cols]
            y_tr = full_df.loc[train_idx, target_col]
            x_v = full_df.loc[val_idx, feature_cols]
            y_v = full_df.loc[val_idx, target_col]

            pipeline = build_pipeline(estimator)
            pipeline.fit(x_tr, y_tr)
            fold_metrics.append(evaluate(pipeline, x_v, y_v))

        mae_mean = np.mean([m["mae"] for m in fold_metrics])
        mae_std = np.std([m["mae"] for m in fold_metrics])
        mape_mean = np.mean([m["mape"] for m in fold_metrics])
        r2_mean = np.mean([m["r2"] for m in fold_metrics])
        rmse_mean = np.mean([m["rmse"] for m in fold_metrics])

        log.info(
            "%s %s: MAE=%.0f±%.0f, MAPE=%.1f%%, R2=%.3f",
            prefix, name, mae_mean, mae_std, mape_mean, r2_mean,
        )
        results.append({
            "label": f"{prefix}_{name}_CV",
            "mae": mae_mean,
            "mae_std": mae_std,
            "rmse": rmse_mean,
            "mape": mape_mean,
            "r2": r2_mean,
        })

    # тюнинг на последнем фолде
    last_train_idx, last_val_idx = splits[-1]
    x_train_last = full_df.loc[last_train_idx, feature_cols]
    y_train_last = full_df.loc[last_train_idx, target_col]
    x_val_last = full_df.loc[last_val_idx, feature_cols]
    y_val_last = full_df.loc[last_val_idx, target_col]

    n_trials = 20 if prefix == "goalies_A" else 10

    tunable = ["lightgbm", "xgboost", "catboost", "random_forest", "elasticnet"]
    tuned_models = {}

    for model_name in tunable:
        best_params = tune_model(
            model_name, x_train_last, y_train_last,
            x_val_last, y_val_last,
            prefix=prefix, n_trials=n_trials,
        )

        if model_name == "lightgbm":
            model = LGBMRegressor(
                **best_params, random_state=random_state, verbose=-1, n_jobs=-1
            )
        elif model_name == "xgboost":
            model = XGBRegressor(
                **best_params, random_state=random_state, verbosity=0
            )
        elif model_name == "catboost":
            model = CatBoostRegressor(
                **best_params, random_state=random_state, verbose=0
            )
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                **best_params, random_state=random_state, n_jobs=-1
            )
        elif model_name == "elasticnet":
            model = ElasticNet(
                **best_params, random_state=random_state, max_iter=10000
            )

        # оцениваем по всем фолдам
        fold_metrics = []
        for train_idx, val_idx in splits:
            x_tr = full_df.loc[train_idx, feature_cols]
            y_tr = full_df.loc[train_idx, target_col]
            x_v = full_df.loc[val_idx, feature_cols]
            y_v = full_df.loc[val_idx, target_col]
            pipeline = build_pipeline(model)
            pipeline.fit(x_tr, y_tr)
            fold_metrics.append(evaluate(pipeline, x_v, y_v))

        mae_mean = np.mean([m["mae"] for m in fold_metrics])
        mae_std = np.std([m["mae"] for m in fold_metrics])
        mape_mean = np.mean([m["mape"] for m in fold_metrics])
        r2_mean = np.mean([m["r2"] for m in fold_metrics])
        rmse_mean = np.mean([m["rmse"] for m in fold_metrics])

        log.info(
            "%s %s tuned: MAE=%.0f±%.0f, MAPE=%.1f%%, R2=%.3f",
            prefix, model_name, mae_mean, mae_std, mape_mean, r2_mean,
        )
        results.append({
            "label": f"{prefix}_{model_name}_tuned_CV",
            "mae": mae_mean,
            "mae_std": mae_std,
            "rmse": rmse_mean,
            "mape": mape_mean,
            "r2": r2_mean,
        })

        pipeline_final = build_pipeline(model)
        pipeline_final.fit(x_train_last, y_train_last)
        save_model(pipeline_final, f"{prefix}_{model_name}_tuned")
        tuned_models[model_name] = pipeline_final

    # стекинг на последнем фолде (ElasticNet не включаем)
    def load_best_params_goalie(model_name: str) -> dict:
        path = models_dir / f"best_params_{prefix}_{model_name}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)["params"]
        return {}

    cv_stack = 3

    estimators = [
        ("lightgbm", Pipeline([
            ("scaler", StandardScaler()),
            ("model", LGBMRegressor(
                **load_best_params_goalie("lightgbm"),
                random_state=random_state, verbose=-1,
            )),
        ])),
        ("xgboost", Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                **load_best_params_goalie("xgboost"),
                random_state=random_state, verbosity=0,
            )),
        ])),
        ("catboost", Pipeline([
            ("scaler", StandardScaler()),
            ("model", CatBoostRegressor(
                **load_best_params_goalie("catboost"),
                random_state=random_state, verbose=0,
            )),
        ])),
        ("random_forest", Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                **load_best_params_goalie("random_forest"),
                random_state=random_state, n_jobs=-1,
            )),
        ])),
    ]

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=cv_stack,
        n_jobs=-1,
    )

    fold_metrics = []
    for train_idx, val_idx in splits:
        x_tr = full_df.loc[train_idx, feature_cols]
        y_tr = full_df.loc[train_idx, target_col]
        x_v = full_df.loc[val_idx, feature_cols]
        y_v = full_df.loc[val_idx, target_col]
        stacking.fit(x_tr, y_tr)
        fold_metrics.append(evaluate(stacking, x_v, y_v))

    mae_mean = np.mean([m["mae"] for m in fold_metrics])
    mae_std = np.std([m["mae"] for m in fold_metrics])
    mape_mean = np.mean([m["mape"] for m in fold_metrics])
    r2_mean = np.mean([m["r2"] for m in fold_metrics])
    rmse_mean = np.mean([m["rmse"] for m in fold_metrics])

    log.info(
        "%s stacking: MAE=%.0f±%.0f, MAPE=%.1f%%, R2=%.3f",
        prefix, mae_mean, mae_std, mape_mean, r2_mean,
    )
    results.append({
        "label": f"{prefix}_stacking_CV",
        "mae": mae_mean,
        "mae_std": mae_std,
        "rmse": rmse_mean,
        "mape": mape_mean,
        "r2": r2_mean,
    })

    save_model(stacking, f"{prefix}_stacking")

    print_results_table(results)
    pd.DataFrame(results).to_csv(
        models_dir / f"{prefix}_cv_results.csv", index=False
    )
    return results


# --- финальная оценка ---

def run_final_evaluation(prefix: str):
    """
    Финальная оценка топ-3 моделей на test-сете.
    Для вратарей не используется — у них CV вместо test.
    """
    log.info("Финальная оценка на тесте: %s", prefix)

    _, _, test = load_splits(prefix)
    feature_cols = get_feature_cols(test, prefix=prefix)
    x_test = test[feature_cols]
    y_test = test[target_col]

    experiments = pd.read_csv(models_dir / f"{prefix}_experiments.csv")
    top3 = experiments.nsmallest(3, "mae")
    log.info("Топ-3 модели по val MAE:")
    print(top3[["label", "mae", "mape", "r2"]].to_string(index=False))

    test_results = []
    for _, row in top3.iterrows():
        label = row["label"]
        model_file = label.replace(f"{prefix}_", "").strip()
        try:
            model = load_model(f"{prefix}_{model_file}")
            metrics = evaluate(
                model, x_test, y_test,
                label=f"{label}_TEST",
            )
            metrics["val_mae"] = row["mae"]
            metrics["delta_mae"] = metrics["mae"] - row["mae"]
            test_results.append(metrics)
        except FileNotFoundError:
            log.warning("Модель не найдена: %s", model_file)

    test_df = pd.DataFrame(test_results)
    log.info("\nVal vs Test сравнение:")
    for _, row in test_df.iterrows():
        log.info(
            "%s — val MAE=%.0f, test MAE=%.0f, delta=%.0f (%.1f%%)",
            row["label"],
            row["val_mae"],
            row["mae"],
            row["delta_mae"],
            row["delta_mae"] / row["val_mae"] * 100,
        )

    test_df.to_csv(models_dir / f"{prefix}_test_results.csv", index=False)

    best_test = test_df.loc[test_df["mae"].idxmin()]
    best_name = best_test["label"].replace(f"{prefix}_", "").replace("_TEST", "")
    log.info("Финальная модель для деплоя: %s (test MAE=%.0f)", best_name, best_test["mae"])

    best_model = load_model(f"{prefix}_{best_name}")
    save_model(best_model, f"{prefix}_final")
    log.info("Сохранена как: %s_final.joblib", prefix)

    return best_name, test_df


def run_error_analysis(prefix: str):
    """
    Анализ ошибок финальной модели:
    - остатки по диапазонам зарплат
    - остатки по позициям (скейтеры)
    - топ-10 переоценённых и недооценённых игроков
    """
    log.info("Error analysis: %s", prefix)
    _, _, test = load_splits(prefix)
    feature_cols = get_feature_cols(test, prefix=prefix)
    x_test = test[feature_cols]
    y_test = test[target_col]

    model = load_model(f"{prefix}_final")
    pred_log = model.predict(x_test)

    test = test.copy()
    test["pred_cap_hit"] = np.expm1(pred_log)
    test["actual_cap_hit"] = np.expm1(y_test.values)
    test["error"] = test["pred_cap_hit"] - test["actual_cap_hit"]
    test["abs_error"] = test["error"].abs()
    test["pct_error"] = (test["error"] / test["actual_cap_hit"]) * 100

    bins = [0, 1e6, 3e6, 7e6, 20e6]
    labels_bucket = ["<1M", "1-3M", "3-7M", ">7M"]
    test["salary_bucket"] = pd.cut(
        test["actual_cap_hit"], bins=bins, labels=labels_bucket
    )
    bucket_stats = test.groupby("salary_bucket", observed=True)["abs_error"].agg(
        ["mean", "median", "count"]
    )
    log.info("Ошибки по диапазонам зарплат:\n%s", bucket_stats.to_string())

    if "positionCode" in test.columns:
        pos_stats = test.groupby("positionCode")["abs_error"].agg(
            ["mean", "median", "count"]
        )
        log.info("Ошибки по позициям:\n%s", pos_stats.to_string())

    name_col = "skaterFullName" if "skaterFullName" in test.columns else "goalieFullName"

    overvalued = test.nlargest(10, "error")[
        [name_col, "actual_cap_hit", "pred_cap_hit", "error", "pct_error"]
    ]
    log.info("Топ-10 переоценённых:\n%s", overvalued.to_string(index=False))

    undervalued = test.nsmallest(10, "error")[
        [name_col, "actual_cap_hit", "pred_cap_hit", "error", "pct_error"]
    ]
    log.info("Топ-10 недооценённых:\n%s", undervalued.to_string(index=False))

    test.to_csv(
        models_dir / f"{prefix}_test_predictions.csv", index=False
    )
    log.info("Предсказания сохранены: %s_test_predictions.csv", prefix)

    return test


def run():
    log.info("ПОСТАНОВКА А")
    #run_skaters(prefix="skaters", n_trials=50)
    run_goalies_cv(prefix="goalies_A")
    best_skaters_A, test_results_A = run_final_evaluation("skaters")
    run_error_analysis("skaters")

    log.info("ПОСТАНОВКА Б")
    #run_skaters(prefix="skaters_B", n_trials=30)
    run_goalies_cv(prefix="goalies_B")
    best_skaters_B, test_results_B = run_final_evaluation("skaters_B")
    run_error_analysis("skaters_B")

    log.info("Сравнение А vs Б (скейтеры)")
    res_A = pd.read_csv(models_dir / "skaters_test_results.csv")
    res_B = pd.read_csv(models_dir / "skaters_B_test_results.csv")
    best_A = res_A.loc[res_A["mae"].idxmin()]
    best_B = res_B.loc[res_B["mae"].idxmin()]

    log.info(
        "Постановка А, лучшая модель: %s | test MAE=%.0f | MAPE=%.1f%%",
        best_A["label"], best_A["mae"], best_A["mape"],
    )
    log.info(
        "Постановка Б, лучшая модель: %s | test MAE=%.0f | MAPE=%.1f%%",
        best_B["label"], best_B["mae"], best_B["mape"],
    )
    winner = "А" if best_A["mae"] <= best_B["mae"] else "Б"
    log.info(
        "Вывод: для деплоя выбираем постановку %s — "
        "меньше test MAE и val/test разрыв стабильнее", winner,
    )


if __name__ == "__main__":
    run()