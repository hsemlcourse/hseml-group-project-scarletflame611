"""
NHL Salary Prediction Dataset Parser
Sources:
  1. NHL Stats API — игровая статистика + биография (официальный API)
  2. Spotrac       — зарплаты / cap hit (Selenium)

Сезоны: 2021-22 → 2025-26
Выход: data/nhl_skaters_YEAR.csv, data/nhl_goalies_YEAR.csv
       data/nhl_skaters_raw.csv, data/nhl_goalies_raw.csv (финальные)
"""

import re
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEASONS = ["20212022", "20222023", "20232024", "20242025", "20252026"]
SEASON_YEARS = [2022, 2023, 2024, 2025, 2026]
OUTPUT_DIR = Path("../data")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

NHL_LEGACY = "https://api.nhle.com/stats/rest/en"
NHL_WEB = "https://api-web.nhle.com"


def make_driver() -> webdriver.Edge:
    opts = EdgeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    driver = webdriver.Edge(
        service=EdgeService(r"/msedgedriver.exe"),
        options=opts,
    )
    driver.set_page_load_timeout(180)
    driver.set_script_timeout(60)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def restart_driver(driver: webdriver.Edge) -> webdriver.Edge:
    try:
        driver.quit()
    except Exception:
        pass
    time.sleep(3)
    return make_driver()


def _nhl_get(endpoint: str, season: str, sort: str, retries: int = 5) -> pd.DataFrame:
    url = f"{NHL_LEGACY}/{endpoint}"
    params = {
        "cayenneExp": f"seasonId={season} and gameTypeId=2",
        "limit": -1,
        "sort": sort,
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            df = pd.json_normalize(r.json().get("data", []))
            df["season"] = season
            return df
        except requests.exceptions.Timeout:
            log.warning(f"  NHL API таймаут (попытка {attempt}/{retries}), ждём {attempt * 10} сек...")
            time.sleep(attempt * 10)
        except requests.exceptions.ConnectionError as e:
            log.warning(f"  NHL API connection error (попытка {attempt}/{retries}): {e}")
            time.sleep(attempt * 10)
        except Exception as e:
            log.warning(f"  NHL endpoint {endpoint} ошибка: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def fetch_player_bio_v1(player_id: int, retries: int = 3) -> dict:
    url = f"{NHL_WEB}/v1/player/{player_id}/landing"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            d = r.json()
            draft = d.get("draftDetails") or {}
            return {
                "playerId": player_id,
                "heightInInches": d.get("heightInInches"),
                "weightInPounds": d.get("weightInPounds"),
                "birthCity": (d.get("birthCity") or {}).get("default", ""),
                "birthCountry": d.get("birthCountry", ""),
                "draftYear": draft.get("year"),
                "draftRound": draft.get("round"),
                "draftOverall": draft.get("pickInRound"),
            }
        except Exception as e:
            log.warning(f"  v1/player/{player_id}: попытка {attempt} — {e}")
            time.sleep(attempt * 2)
    return {"playerId": player_id}


def build_v1_cache(all_ids: set) -> pd.DataFrame:
    """Загружаем v1 биографии один раз для всех игроков."""
    log.info(f"v1 API: загружаем биографии для {len(all_ids)} уникальных игроков...")
    records = []
    for i, pid in enumerate(all_ids):
        records.append(fetch_player_bio_v1(int(pid)))
        if i % 100 == 0 and i > 0:
            log.info(f"  {i}/{len(all_ids)}...")
            time.sleep(1)
        else:
            time.sleep(0.1)
    df = pd.DataFrame(records)
    log.info(f"v1 API: кеш готов, {len(df)} записей")
    return df


def enrich_from_cache(df: pd.DataFrame, v1_cache: pd.DataFrame) -> pd.DataFrame:
    if "playerId" not in df.columns or v1_cache.empty:
        return df
    extra_cols = [c for c in v1_cache.columns if c != "playerId" and c not in df.columns]
    if extra_cols:
        df = df.merge(v1_cache[["playerId"] + extra_cols], on="playerId", how="left")
        log.info(f"  v1 cache: добавлены колонки {extra_cols}")
    return df


def build_nhl_skaters(season: str) -> pd.DataFrame:
    log.info(f"NHL API → скейтеры {season}")
    base = _nhl_get("skater/summary", season, "points")
    log.info(f"  → {len(base)} игроков")
    time.sleep(0.5)

    merge_key = ["playerId", "season"]
    endpoints = [
        ("skater/realtime", "hits"),
        ("skater/faceoffpercentages", "totalFaceoffs"),
        ("skater/powerplay", "ppPoints"),
        ("skater/penaltyShots", "goals"),
        ("skater/shootout", "wins"),
    ]

    for endpoint, sort in endpoints:
        try:
            extra = _nhl_get(endpoint, season, sort)
            if not extra.empty:
                drop_cols = [c for c in extra.columns if c in base.columns and c not in merge_key]
                extra = extra.drop(columns=drop_cols, errors="ignore")
                base = base.merge(extra, on=merge_key, how="left")
                log.info(f"  + {endpoint}: merged")
        except Exception as e:
            log.warning(f"  NHL endpoint {endpoint} ошибка: {e}")
        time.sleep(0.5)

    return base


def build_nhl_goalies(season: str) -> pd.DataFrame:
    log.info(f"NHL API → вратари {season}")
    base = _nhl_get("goalie/summary", season, "wins")
    log.info(f"  → {len(base)} вратарей")
    time.sleep(0.5)

    merge_key = ["playerId", "season"]
    endpoints = [
        ("goalie/advanced", "qualityStart"),
        ("goalie/shootout", "wins"),
    ]

    for endpoint, sort in endpoints:
        try:
            extra = _nhl_get(endpoint, season, sort)
            if not extra.empty:
                drop_cols = [c for c in extra.columns if c in base.columns and c not in merge_key]
                extra = extra.drop(columns=drop_cols, errors="ignore")
                base = base.merge(extra, on=merge_key, how="left")
                log.info(f"  + {endpoint}: merged")
        except Exception as e:
            log.warning(f"  NHL endpoint {endpoint} ошибка: {e}")
        time.sleep(0.5)

    return base


def fetch_spotrac_salaries(driver: webdriver.Edge, year: int) -> pd.DataFrame:
    log.info(f"Spotrac → зарплаты {year - 1}-{str(year)[-2:]}")
    url = f"https://www.spotrac.com/nhl/rankings/player/_/year/{year}/sort/cap_total"
    try:
        try:
            driver.get(url)
        except Exception as e:
            log.warning(f"  Spotrac page load timeout, пробуем всё равно: {e}")

        try:
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "ul.list-group li.list-group-item"))
            )
        except Exception:
            log.warning("  Spotrac: таймаут ожидания элементов, пробуем всё равно")
        time.sleep(3)

        html = driver.page_source
        soup = BeautifulSoup(html, "html5lib")

        rows = []
        items = soup.select("ul.list-group li.list-group-item")
        log.info(f"  Найдено li элементов: {len(items)}")

        for item in items:
            name_tag = item.select_one("a.link")
            if not name_tag:
                continue
            name = name_tag.get_text(strip=True)

            small = item.select_one("small")
            team, pos = "", ""
            if small:
                parts = small.get_text(" ", strip=True).split(",")
                if len(parts) >= 2:
                    team = parts[0].strip()
                    pos = parts[1].strip()

            salary_tag = item.select_one("span.medium")
            cap_hit = None
            if salary_tag:
                raw = salary_tag.get_text(strip=True).replace("$", "").replace(",", "").strip()
                try:
                    cap_hit = float(raw)
                except ValueError:
                    cap_hit = None

            rows.append({
                "player_name_cw": name,
                "team_cw": team,
                "position_cw": pos,
                "cap_hit": cap_hit,
                "season_year": year,
            })

        df = pd.DataFrame(rows)
        log.info(f"  → {len(df)} игроков")
        return df

    except Exception as e:
        log.error(f"  Spotrac ошибка: {e}")
        return pd.DataFrame()


_TRANSLIT = str.maketrans(
    "áéíóúäöüčšžřýěňÁÉÍÓÚÄÖÜČŠŽŘÝĚŇ",
    "aeiouaoucsrzyenAEIOUAOUCSRZYEN",
)


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    n = n.translate(_TRANSLIT)
    n = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", n)
    n = re.sub(r"[^a-z\s]", "", n)
    return n.strip()


def build_dataset():
    driver = make_driver()
    log.info("Edge драйвер запущен")

    # Собираем все уникальные ID игроков по всем сезонам
    log.info("Собираем список всех игроков...")
    all_ids = set()
    for season in SEASONS:
        df_sk = _nhl_get("skater/summary", season, "points")
        if not df_sk.empty and "playerId" in df_sk.columns:
            all_ids.update(df_sk["playerId"].astype(int).tolist())
        df_go = _nhl_get("goalie/summary", season, "wins")
        if not df_go.empty and "playerId" in df_go.columns:
            all_ids.update(df_go["playerId"].astype(int).tolist())
        time.sleep(0.3)
    log.info(f"Всего уникальных игроков: {len(all_ids)}")

    # Загружаем v1 биографии один раз для всех
    v1_cache = build_v1_cache(all_ids)

    try:
        for season, year in zip(SEASONS, SEASON_YEARS):
            log.info(f"\n{'=' * 60}")
            log.info(f"  СЕЗОН {season}  ({year - 1}-{str(year)[-2:]})")
            log.info(f"{'=' * 60}")

            path_sk = OUTPUT_DIR / f"nhl_skaters_{year}.csv"
            path_go = OUTPUT_DIR / f"nhl_goalies_{year}.csv"
            if path_sk.exists() and path_go.exists():
                log.info(f"  Сезон {year} уже сохранён, пропускаем")
                continue

            try:
                sk = build_nhl_skaters(season)
                sk["season_year"] = year
            except Exception as e:
                log.error(f"NHL skaters: {e}")
                sk = pd.DataFrame()

            try:
                go = build_nhl_goalies(season)
                go["season_year"] = year
            except Exception as e:
                log.error(f"NHL goalies: {e}")
                go = pd.DataFrame()

            time.sleep(1.5)

            try:
                sal = fetch_spotrac_salaries(driver, year)
            except Exception as e:
                log.error(f"  Spotrac упал: {e}, перезапускаем драйвер")
                driver = restart_driver(driver)
                try:
                    sal = fetch_spotrac_salaries(driver, year)
                except Exception:
                    sal = pd.DataFrame()
            time.sleep(2.0)

            if not sk.empty:
                name_col = next(
                    (c for c in ["skaterFullName", "playerName"] if c in sk.columns), None
                )
                if name_col:
                    sk["name_key"] = sk[name_col].apply(normalize_name)

                sk = enrich_from_cache(sk, v1_cache)

                if not sal.empty and "player_name_cw" in sal.columns:
                    sal["name_key"] = sal["player_name_cw"].apply(normalize_name)
                    sal_slim = sal.drop(
                        columns=[c for c in sal.columns if c in sk.columns and c != "name_key"],
                        errors="ignore",
                    )
                    sk = sk.merge(sal_slim, on="name_key", how="left")
                    matched = sk["cap_hit"].notna().sum() if "cap_hit" in sk.columns else 0
                    log.info(f"  Spotrac join: {matched}/{len(sk)} игроков с зарплатой")

                sk.dropna(axis=1, how="all", inplace=True)
                sk.to_csv(path_sk, index=False, encoding="utf-8-sig")
                log.info(f"  Скейтеры сохранены: {path_sk} ({sk.shape})")

            if not go.empty:
                name_col_g = next(
                    (c for c in ["goalieFullName", "playerName"] if c in go.columns), None
                )
                if name_col_g:
                    go["name_key"] = go[name_col_g].apply(normalize_name)

                go = enrich_from_cache(go, v1_cache)

                if not sal.empty and "player_name_cw" in sal.columns:
                    sal["name_key"] = sal["player_name_cw"].apply(normalize_name)
                    sal_slim = sal.drop(
                        columns=[c for c in sal.columns if c in go.columns and c != "name_key"],
                        errors="ignore",
                    )
                    go = go.merge(sal_slim, on="name_key", how="left")
                    matched = go["cap_hit"].notna().sum() if "cap_hit" in go.columns else 0
                    log.info(f"  Spotrac join вратари: {matched}/{len(go)} с зарплатой")

                go.dropna(axis=1, how="all", inplace=True)
                go.to_csv(path_go, index=False, encoding="utf-8-sig")
                log.info(f"  Вратари сохранены: {path_go} ({go.shape})")

            log.info(f"Сезон {season} готов. Пауза 3 сек...\n")
            time.sleep(3.0)

    finally:
        driver.quit()
        log.info("Edge драйвер закрыт")

    _merge_all_seasons()


def _merge_all_seasons():
    sk_files = sorted(OUTPUT_DIR.glob("nhl_skaters_2*.csv"))
    go_files = sorted(OUTPUT_DIR.glob("nhl_goalies_2*.csv"))

    if sk_files:
        df_sk = pd.concat([pd.read_csv(f) for f in sk_files], ignore_index=True)
        df_sk.dropna(axis=1, how="all", inplace=True)
        path = OUTPUT_DIR / "nhl_skaters_raw.csv"
        df_sk.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"\nИтого скейтеры: {df_sk.shape[0]} строк × {df_sk.shape[1]} колонок → {path}")

    if go_files:
        df_go = pd.concat([pd.read_csv(f) for f in go_files], ignore_index=True)
        df_go.dropna(axis=1, how="all", inplace=True)
        path = OUTPUT_DIR / "nhl_goalies_raw.csv"
        df_go.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"Итого вратари: {df_go.shape[0]} строк × {df_go.shape[1]} колонок → {path}")

    log.info("\nПарсинг завершён.")


if __name__ == "__main__":
    build_dataset()
