"""
Пайплайн предобработки данных для предсказания зарплат игроков NHL.

Скейтеры: data/raw/nhl_skaters_raw.csv -> data/processed/ -> data/features/
Вратари:  data/raw/nhl_goalies_raw.csv -> data/processed/ -> data/features/
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

random_state = 42
min_games_played = 10
target_col = "cap_hit"
log_target_col = "log_cap_hit"

# постановка А
train_seasons = [2022, 2023, 2024]
val_seasons = [2025]
test_seasons = [2026]

# постановка Б: предсказываем контракт N+1
train_seasons_B = [2022, 2023]
val_seasons_B = [2024]
test_seasons_B = [2025]

raw_dir = Path("../data/raw")
processed_dir = Path("../data/processed")
features_dir = Path("../data/features")

# колонки из Spotrac, пришедшие вместе с таргетом убираем до обучения
leakage_cols = ["player_name_cw", "team_cw", "position_cw", "name_key"]

id_cols_skaters = ["playerId", "skaterFullName", "lastName", "season", "seasonId", "season_year"]
id_cols_goalies = ["playerId", "goalieFullName", "lastName", "season", "seasonId", "season_year"]


# общие утилиты


def setup_dirs():
    for d in [processed_dir, features_dir]:
        d.mkdir(parents=True, exist_ok=True)


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def make_temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разбивка по сезонам для исключения утечки данных.
    Cap hit назначается до начала сезона, поэтому сплит строго хронологический:
    train < val < test.
    """
    train = df[df["season_year"].isin(train_seasons)].copy()
    val = df[df["season_year"].isin(val_seasons)].copy()
    test = df[df["season_year"].isin(test_seasons)].copy()
    log.info("Разбивка: train=%d, val=%d, test=%d строк", len(train), len(val), len(test))
    return train, val, test


def make_target_shift(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["playerId", "season_year"]).copy()

    df["cap_hit_next"] = df.groupby("playerId")["cap_hit"].shift(-1)

    df["salary_changed"] = ((df["cap_hit_next"] - df["cap_hit"]).abs() / df["cap_hit"]) > 0.05

    # убрали ограничение по году, оставляем всех у кого есть N+1
    df = df[df["cap_hit_next"].notna() & df["salary_changed"]].copy()

    df[target_col] = df["cap_hit_next"]
    df[log_target_col] = np.log1p(df["cap_hit_next"])

    log.info("Постановка Б: %d игроков со сменой контракта", len(df))
    return df


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    По данным EDA: skew(cap_hit) = 0.90 у скейтеров, 0.99 у вратарей.
    После логарифмирования skew падает до -0.01 / -0.04.
    При оценке моделей предсказания переводим обратно через expm1.
    """
    df[log_target_col] = np.log1p(df[target_col])
    return df


def report_missing(df: pd.DataFrame, label: str):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        log.info("%s: пропусков нет", label)
    else:
        log.info("%s, пропуски:\n%s", label, missing.to_string())


# очистка скейтеров


def clean_skaters(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)

    # удаляем строки без таргета
    df = df.dropna(subset=[target_col])
    log.info(
        "Удалено %d строк без cap_hit (%.1f%%)",
        initial - len(df),
        (initial - len(df)) / initial * 100,
    )

    # убираем аномально низкие значения
    before_threshold = len(df)
    df = df[df[target_col] >= 500_000]
    log.info("Удалено %d строк с cap_hit < 500K", before_threshold - len(df))

    # фильтр по количеству игр
    df = df[df["gamesPlayed"] >= min_games_played]
    log.info("После фильтра по играм: осталось %d строк", len(df))

    # удаляем точные дубликаты по игроку и сезону
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["playerId", "season_year"])
    log.info("Удалено %d дублирующихся строк", before_dedup - len(df))

    # типы
    df["season_year"] = df["season_year"].astype(int)
    df["gamesPlayed"] = df["gamesPlayed"].astype(int)
    df[target_col] = df[target_col].astype(float)

    # вбрасывания: NaN у игроков не-центров заполняем нулями
    faceoff_cols = [c for c in df.columns if "faceoff" in c.lower() or "Faceoff" in c]
    df[faceoff_cols] = df[faceoff_cols].fillna(0)

    # Corsi отсутствует только в сезоне 2022 (старый API не отдавал) заполняем нулями
    for c in ["totalShotAttempts", "shotAttemptsBlocked"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # пустые ворота и промахи: NaN у игроков с малым льдом заполняем нулями
    en_cols = [c for c in df.columns if "emptyNet" in c or "missedShot" in c]
    df[en_cols] = df[en_cols].fillna(0)

    # большинство: NaN у игроков без времени в большинстве заполняем нулями
    pp_cols = [c for c in df.columns if c.startswith("pp")]
    df[pp_cols] = df[pp_cols].fillna(0)

    report_missing(df, "скейтеры после очистки")
    return df


# feature engineering скейтеров


def engineer_skater_features(df: pd.DataFrame) -> pd.DataFrame:
    # возраст: большинство игроков задрафтованы в 18 лет,
    # поэтому draftYear + 18 даёт приближение года рождения
    # недрафтованным игрокам заполняем медианой по сезону
    # к cp2 мб поищу откуда брать
    df["age"] = df["season_year"] - df["draftYear"] + 18
    median_age = df.groupby("season_year")["age"].transform("median")
    df["age"] = df["age"].fillna(median_age)

    df["is_undrafted"] = df["draftYear"].isna().astype(int)

    # играл ли за несколько команд в сезоне
    df["is_multi_team"] = df["teamAbbrevs"].str.contains(",").astype(int)

    # нападающий или защитник
    df["is_forward"] = (df["positionCode"].isin(["C", "L", "R"])).astype(int)

    # метрики per-60 нормированные по реальному льду
    toi_per_game = df["timeOnIcePerGame"].replace(0, np.nan)
    games = df["gamesPlayed"].replace(0, np.nan)
    total_toi_h = toi_per_game * games / 3600

    df["goals_per_60"] = df["goals"] / total_toi_h * 60
    df["assists_per_60"] = df["assists"] / total_toi_h * 60
    df["points_per_60"] = df["points"] / total_toi_h * 60

    # комбинированный признак ценности пика на драфте:
    # draftOverall содержит номер пика внутри раунда (1–36), а не overall pick.
    # по EDA: max round = 7, max pick in round = 36, max draft_value = 252.
    # недрафтованным присваиваем 302 (252 + 50), это штрафное значение
    df["draft_value"] = (df["draftRound"] - 1) * 36 + df["draftOverall"]
    df["draft_value"] = df["draft_value"].fillna(302)

    # индекс физических данных
    df["size_index"] = df["heightInInches"] * df["weightInPounds"]

    # оставшиеся числовые NaN заполняем нулями
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def add_lag_features_skaters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["playerId", "season_year"]).copy()

    lag_cols = [
        "goals",
        "assists",
        "points",
        "timeOnIcePerGame",
        "points_per_60",
        "goals_per_60",
        "assists_per_60",
        "hits",
        "blockedShots",
        "plusMinus",
        "ppPoints",
        "cap_hit",
    ]

    for col in lag_cols:
        if col not in df.columns:
            continue
        df[f"{col}_lag1"] = df.groupby("playerId")[col].shift(1)
        df[f"{col}_delta"] = df[col] - df[f"{col}_lag1"]

    # заполняем NaN у игроков без предыдущего сезона нулями
    lag_created = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_delta")]
    df[lag_created] = df[lag_created].fillna(0)

    log.info("Лаговые признаки скейтеры: добавлено %d колонок", len(lag_created))
    return df


def add_lag_features_goalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["playerId", "season_year"]).copy()

    lag_cols = [
        "savePct",
        "goalsAgainstAverage",
        "wins",
        "shutouts",
        "gamesStarted",
        "gsaa_proxy",
        "win_rate",
        "cap_hit",
    ]

    for col in lag_cols:
        if col not in df.columns:
            continue
        df[f"{col}_lag1"] = df.groupby("playerId")[col].shift(1)
        df[f"{col}_delta"] = df[col] - df[f"{col}_lag1"]

    lag_created = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_delta")]
    df[lag_created] = df[lag_created].fillna(0)

    log.info("Лаговые признаки вратари: добавлено %d колонок", len(lag_created))
    return df


def add_country_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Группируем страны рождения в 7 категорий.
    Топ-6 хоккейных стран отдельно, остальные в OTHER.
    """
    hockey_countries = ["CAN", "USA", "RUS", "SWE", "FIN", "CZE"]

    df["country_group"] = df["birthCountry"].apply(
        lambda x: x if x in hockey_countries else "OTHER"
    )
    dummies = pd.get_dummies(
        df["country_group"],
        prefix="country",
        dtype=int,
    )
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=["country_group"])
    created = [c for c in dummies.columns]
    log.info("Страны: %s", df["birthCountry"].value_counts().head(8).to_dict())
    log.info("Добавлены колонки: %s", created)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # нелинейность возраста
    df["age_squared"] = df["age"] ** 2
    df["age_x_points_per_60"] = df["age"] * df["points_per_60"]

    # признаки нападающих
    df["points_per_60_x_forward"] = df["points_per_60"] * df["is_forward"]
    df["goals_per_60_x_forward"] = df["goals_per_60"] * df["is_forward"]

    # признаки защитников
    is_defense = 1 - df["is_forward"]
    df["hits_per_60_x_defense"] = df["hitsPer60"] * is_defense
    df["blocked_per_60_x_defense"] = df["blockedShotsPer60"] * is_defense
    df["plusminus_x_defense"] = df["plusMinus"] * is_defense

    created = [
        "age_squared",
        "age_x_points_per_60",
        "points_per_60_x_forward",
        "goals_per_60_x_forward",
        "hits_per_60_x_defense",
        "blocked_per_60_x_defense",
        "plusminus_x_defense",
    ]
    log.info("Interaction features: добавлено %d колонок", len(created))
    return df


def prepare_skaters() -> dict[str, pd.DataFrame]:
    log.info("Загружаем сырые данные скейтеров")
    df = pd.read_csv(raw_dir / "nhl_skaters_raw.csv")
    log.info("Загружено %d строк и %d колонок", *df.shape)

    df = drop_leakage(df)
    df = clean_skaters(df)
    df = engineer_skater_features(df)
    df = add_country_features(df)
    df = add_interaction_features(df)
    df = add_log_target(df)

    df.to_csv(processed_dir / "skaters_processed.csv", index=False, encoding="utf-8-sig")
    log.info("Сохранены обработанные скейтеры: %s", processed_dir / "skaters_processed.csv")

    train, val, test = make_temporal_split(df)
    train.to_csv(features_dir / "skaters_train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(features_dir / "skaters_val.csv", index=False, encoding="utf-8-sig")
    test.to_csv(features_dir / "skaters_test.csv", index=False, encoding="utf-8-sig")
    log.info("Сплиты скейтеров сохранены в %s", features_dir)

    return {"train": train, "val": val, "test": test}


# очистка вратарей


def clean_goalies(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)

    df = df.dropna(subset=[target_col])
    log.info(
        "Удалено %d вратарей без cap_hit (%.1f%%)",
        initial - len(df),
        (initial - len(df)) / initial * 100,
    )

    df = df[df["gamesPlayed"] >= min_games_played]
    log.info("После фильтра по играм: осталось %d вратарей", len(df))

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["playerId", "season_year"])
    log.info("Удалено %d дублирующихся строк вратарей", before_dedup - len(df))

    df["season_year"] = df["season_year"].astype(int)
    df[target_col] = df[target_col].astype(float)

    # qualityStartsPct и completeGamePct: NaN при 0 стартах заполняем нулями
    for c in ["qualityStartsPct", "completeGamePct"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # savePct: заполняем медианой
    if "savePct" in df.columns:
        df["savePct"] = df["savePct"].fillna(df["savePct"].median())

    report_missing(df, "вратари после очистки")
    return df


# feature engineering вратарей


def engineer_goalie_features(df: pd.DataFrame) -> pd.DataFrame:
    df["age"] = df["season_year"] - df["draftYear"] + 18
    median_age = df.groupby("season_year")["age"].transform("median")
    df["age"] = df["age"].fillna(median_age)

    df["is_undrafted"] = df["draftYear"].isna().astype(int)
    df["is_multi_team"] = df["teamAbbrevs"].str.contains(",").astype(int)

    # доля матчей, в которых вратарь выходил в старте
    df["starter_ratio"] = df["gamesStarted"] / df["gamesPlayed"].replace(0, np.nan)
    df["starter_ratio"] = df["starter_ratio"].fillna(0)

    # прокси GSAA: разница между реальным и средним по лиге sv%,
    # умноженная на количество бросков, показывает сколько голов вратарь сэкономил.
    # league_avg_svpct = 0.8933 по EDA
    league_avg_svpct = 0.8933
    df["gsaa_proxy"] = (df["savePct"] - league_avg_svpct) * df["shotsAgainst"]

    # процент побед
    df["win_rate"] = df["wins"] / df["gamesPlayed"].replace(0, np.nan)
    df["win_rate"] = df["win_rate"].fillna(0)

    # аналогично скейтерам: draft_value как комбинированный признак
    df["draft_value"] = (df["draftRound"] - 1) * 36 + df["draftOverall"]
    df["draft_value"] = df["draft_value"].fillna(302)
    df["size_index"] = df["heightInInches"] * df["weightInPounds"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def prepare_goalies() -> dict[str, pd.DataFrame]:
    log.info("Загружаем сырые данные вратарей")
    df = pd.read_csv(raw_dir / "nhl_goalies_raw.csv")
    log.info("Загружено %d строк и %d колонок", *df.shape)

    df = drop_leakage(df)
    df = clean_goalies(df)
    df = engineer_goalie_features(df)
    df = add_lag_features_goalies(df)
    df = add_country_features(df)
    df = add_log_target(df)

    df.to_csv(processed_dir / "goalies_processed.csv", index=False, encoding="utf-8-sig")
    log.info("Сохранены обработанные вратари: %s", processed_dir / "goalies_processed.csv")

    train, val, test = make_temporal_split(df)
    train.to_csv(features_dir / "goalies_train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(features_dir / "goalies_val.csv", index=False, encoding="utf-8-sig")
    test.to_csv(features_dir / "goalies_test.csv", index=False, encoding="utf-8-sig")
    log.info("Сплиты вратарей сохранены в %s", features_dir)

    return {"train": train, "val": val, "test": test}


def prepare_skaters_B() -> dict[str, pd.DataFrame]:
    log.info("Постановка Б: скейтеры")
    df = pd.read_csv(raw_dir / "nhl_skaters_raw.csv")

    df = drop_leakage(df)
    df = clean_skaters(df)
    df = engineer_skater_features(df)
    df = add_lag_features_skaters(df)
    df = add_country_features(df)
    df = add_interaction_features(df)
    # применяем сдвиг и фильтрацию
    df = make_target_shift(df)

    # сплиты по season_year статистики
    train = df[df["season_year"].isin(train_seasons_B)].copy()
    val = df[df["season_year"].isin(val_seasons_B)].copy()
    test = df[df["season_year"].isin(test_seasons_B)].copy()
    log.info("Б скейтеры: train=%d, val=%d, test=%d", len(train), len(val), len(test))

    train.to_csv(features_dir / "skaters_B_train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(features_dir / "skaters_B_val.csv", index=False, encoding="utf-8-sig")
    test.to_csv(features_dir / "skaters_B_test.csv", index=False, encoding="utf-8-sig")

    return {"train": train, "val": val, "test": test}


def prepare_goalies_B() -> dict[str, pd.DataFrame]:
    log.info("Постановка Б: вратари")
    df = pd.read_csv(raw_dir / "nhl_goalies_raw.csv")

    df = drop_leakage(df)
    df = clean_goalies(df)
    df = engineer_goalie_features(df)
    df = add_lag_features_goalies(df)
    df = add_country_features(df)
    df = make_target_shift(df)

    train = df[df["season_year"].isin(train_seasons_B)].copy()
    val = df[df["season_year"].isin(val_seasons_B)].copy()
    test = df[df["season_year"].isin(test_seasons_B)].copy()
    log.info("Б вратари: train=%d, val=%d, test=%d", len(train), len(val), len(test))

    train.to_csv(features_dir / "goalies_B_train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(features_dir / "goalies_B_val.csv", index=False, encoding="utf-8-sig")
    test.to_csv(features_dir / "goalies_B_test.csv", index=False, encoding="utf-8-sig")

    return {"train": train, "val": val, "test": test}


def run():
    setup_dirs()
    log.info("ПОСТАНОВКА А: текущий cap hit")
    skater_splits_A = prepare_skaters()
    goalie_splits_A = prepare_goalies()

    log.info("ПОСТАНОВКА Б: будущий контракт")
    skater_splits_B = prepare_skaters_B()
    goalie_splits_B = prepare_goalies_B()

    _validate_splits()

    return skater_splits_A, goalie_splits_A, skater_splits_B, goalie_splits_B


def _validate_splits():
    """
    Проверяет что сплиты не пересекаются по сезонам и таргет корректен.
    """
    log.info("Валидация сплитов...")
    checks = [
        ("skaters_A", "skaters"),
        ("skaters_B", "skaters_B"),
        ("goalies_A", "goalies"),
        ("goalies_B", "goalies_B"),
    ]
    for label, prefix in checks:
        train_path = features_dir / f"{prefix}_train.csv"
        val_path = features_dir / f"{prefix}_val.csv"
        test_path = features_dir / f"{prefix}_test.csv"

        if not all(p.exists() for p in [train_path, val_path, test_path]):
            log.warning("%s: файлы не найдены", label)
            continue

        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)

        # сезоны не пересекаются
        train_seasons_set = set(train["season_year"].unique())
        val_seasons_set = set(val["season_year"].unique())
        test_seasons_set = set(test["season_year"].unique())

        overlap_tv = train_seasons_set & val_seasons_set
        overlap_vt = val_seasons_set & test_seasons_set
        overlap_tt = train_seasons_set & test_seasons_set

        if overlap_tv or overlap_vt or overlap_tt:
            log.error(
                "%s: ПЕРЕСЕЧЕНИЕ! train с val=%s, val с test=%s, train с test=%s",
                label,
                overlap_tv,
                overlap_vt,
                overlap_tt,
            )
        else:
            log.info(
                "%s: сезоны OK: train%s val%s test%s",
                label,
                sorted(train_seasons_set),
                sorted(val_seasons_set),
                sorted(test_seasons_set),
            )

        # таргет не пустой и в разумном диапазоне
        for split_name, df in [("train", train), ("val", val), ("test", test)]:
            if "log_cap_hit" not in df.columns:
                log.error("%s %s: нет колонки log_cap_hit", label, split_name)
                continue
            nulls = df["log_cap_hit"].isna().sum()
            if nulls > 0:
                log.error("%s %s: %d NaN в log_cap_hit", label, split_name, nulls)
            cap_min = np.expm1(df["log_cap_hit"].min()) / 1e6
            cap_max = np.expm1(df["log_cap_hit"].max()) / 1e6
            log.info(
                "%s %s: %d строк, cap_hit=[%.2fM, %.2fM]",
                label,
                split_name,
                len(df),
                cap_min,
                cap_max,
            )

        # постановка Б: проверяем что cap_hit_next > 0
        if "B" in label and "cap_hit_next" in train.columns:
            for split_name, df in [("train", train), ("val", val), ("test", test)]:
                bad = (df["cap_hit_next"] < 500_000).sum()
                if bad > 0:
                    log.warning("%s %s: %d строк с cap_hit_next < 500K", label, split_name, bad)

    log.info("Валидация завершена")


# -----Препроцессинг одного игрока для API------

SKATER_LAG_COLS = [
    "goals",
    "assists",
    "points",
    "timeOnIcePerGame",
    "points_per_60",
    "goals_per_60",
    "assists_per_60",
    "hits",
    "blockedShots",
    "plusMinus",
    "ppPoints",
    "cap_hit",
]

GOALIE_LAG_COLS = [
    "savePct",
    "goalsAgainstAverage",
    "wins",
    "shutouts",
    "gamesStarted",
    "gsaa_proxy",
    "win_rate",
    "cap_hit",
]

HOCKEY_COUNTRIES = ["CAN", "USA", "RUS", "SWE", "FIN", "CZE"]


def _add_country_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Воспроизводит add_country_features для одной строки."""
    country = df["birthCountry"].iloc[0] if "birthCountry" in df.columns else ""
    group = country if country in HOCKEY_COUNTRIES else "OTHER"
    for c in HOCKEY_COUNTRIES + ["OTHER"]:
        df[f"country_{c}"] = 1 if group == c else 0
    return df


def engineer_one_skater(
    raw: dict,
    prev_season: dict | None = None,
) -> pd.DataFrame:
    """
    Препроцессинг одного скейтера для предсказания через API.

    raw: словарь с полями игрока (из SkaterInput)
    prev_season: опциональная статистика прошлого сезона для лаговых фич.
                 Если None — лаговые фичи заполняются нулями.

    Возвращает pd.DataFrame с одной строкой, готовой к model.predict().
    """
    df = pd.DataFrame([raw])
    defaults = {
        "gamesPlayed": 60,
        "goals": 0,
        "assists": 0,
        "points": 0,
        "plusMinus": 0,
        "timeOnIcePerGame": 900,
        "ppGoals": 0,
        "ppPoints": 0,
        "ppTimeOnIcePerGame": 0,
        "hits": 0,
        "blockedShots": 0,
        "hitsPer60": 0,
        "blockedShotsPer60": 0,
        "takeawaysPer60": 0,
        "heightInInches": 73,
        "weightInPounds": 200,
        "season_year": 2026,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    df["pointsPerGame"] = (df["points"] / df["gamesPlayed"].replace(0, np.nan)).fillna(0)

    games = df["gamesPlayed"].replace(0, np.nan).fillna(60)
    toi_h = (df["timeOnIcePerGame"].replace(0, np.nan) * games / 3600).fillna(0)
    toi_h = toi_h.replace(0, np.nan)
    df["goals_per_60"] = (df["goals"] / toi_h * 60).fillna(0)
    df["assists_per_60"] = (df["assists"] / toi_h * 60).fillna(0)
    df["points_per_60"] = (df["points"] / toi_h * 60).fillna(0)

    if "draftYear" in df.columns and df["draftYear"].notna().all():
        df["age"] = df["season_year"] - df["draftYear"] + 18
        df["is_undrafted"] = 0
    else:
        df["age"] = 28
        df["is_undrafted"] = 1
        df["draftYear"] = np.nan
        df["draftRound"] = np.nan
        df["draftOverall"] = np.nan

    if df["draftRound"].notna().all() and df["draftOverall"].notna().all():
        df["draft_value"] = (df["draftRound"] - 1) * 36 + df["draftOverall"]
    else:
        df["draft_value"] = 302

    df["size_index"] = df.get("heightInInches", 73) * df.get("weightInPounds", 200)

    pos = raw.get("positionCode", "C")
    df["is_forward"] = 1 if pos in ["C", "L", "R"] else 0
    df["is_multi_team"] = 0  # для нового игрока не знаем

    df["age_squared"] = df["age"] ** 2
    df["age_x_points_per_60"] = df["age"] * df["points_per_60"]
    df["points_per_60_x_forward"] = df["points_per_60"] * df["is_forward"]
    df["goals_per_60_x_forward"] = df["goals_per_60"] * df["is_forward"]
    is_defense = 1 - df["is_forward"]
    df["hits_per_60_x_defense"] = df.get("hitsPer60", 0) * is_defense
    df["blocked_per_60_x_defense"] = df.get("blockedShotsPer60", 0) * is_defense
    df["plusminus_x_defense"] = df["plusMinus"] * is_defense

    df = _add_country_dummies(df)

    for col in SKATER_LAG_COLS:
        if prev_season and col in prev_season and prev_season[col] is not None:
            df[f"{col}_lag1"] = prev_season[col]
            current = raw.get(col, 0) or 0
            df[f"{col}_delta"] = current - prev_season[col]
        else:
            df[f"{col}_lag1"] = 0
            df[f"{col}_delta"] = 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def engineer_one_goalie(
    raw: dict,
    prev_season: dict | None = None,
) -> pd.DataFrame:
    """
    Препроцессинг одного вратаря для предсказания через API.
    """
    df = pd.DataFrame([raw])
    defaults = {
        "gamesPlayed": 40,
        "gamesStarted": 35,
        "wins": 0,
        "losses": 0,
        "otLosses": 0,
        "savePct": 0.910,
        "goalsAgainst": 0,
        "goalsAgainstAverage": 2.8,
        "shotsAgainst": 1100,
        "shutouts": 0,
        "qualityStart": 0,
        "qualityStartsPct": 0.0,
        "completeGamePct": 0.0,
        "goalsFor": 0,
        "goalsForAverage": 0.0,
        "shotsAgainstPer60": 0.0,
        "incompleteGames": 0,
        "heightInInches": 74,
        "weightInPounds": 195,
        "season_year": 2026,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    df["starter_ratio"] = (df.get("gamesStarted", 0) / df["gamesPlayed"].replace(0, np.nan)).fillna(
        0
    )

    league_avg_svpct = 0.8933
    df["gsaa_proxy"] = (df.get("savePct", league_avg_svpct) - league_avg_svpct) * df.get(
        "shotsAgainst", 0
    )

    df["win_rate"] = (df.get("wins", 0) / df["gamesPlayed"].replace(0, np.nan)).fillna(0)

    if "draftYear" in df.columns and df["draftYear"].notna().all():
        df["age"] = df["season_year"] - df["draftYear"] + 18
        df["is_undrafted"] = 0
    else:
        df["age"] = 28
        df["is_undrafted"] = 1
        df["draftYear"] = np.nan
        df["draftRound"] = np.nan
        df["draftOverall"] = np.nan

    if df["draftRound"].notna().all() and df["draftOverall"].notna().all():
        df["draft_value"] = (df["draftRound"] - 1) * 36 + df["draftOverall"]
    else:
        df["draft_value"] = 302

    df["size_index"] = df.get("heightInInches", 73) * df.get("weightInPounds", 195)
    df["is_multi_team"] = 0
    df["age_squared"] = df["age"] ** 2

    df = _add_country_dummies(df)

    for col in GOALIE_LAG_COLS:
        if prev_season and col in prev_season and prev_season[col] is not None:
            df[f"{col}_lag1"] = prev_season[col]
            current = raw.get(col, 0) or 0
            df[f"{col}_delta"] = current - prev_season[col]
        else:
            df[f"{col}_lag1"] = 0
            df[f"{col}_delta"] = 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


if __name__ == "__main__":
    run()
