"""
Режим поиска реального игрока из датасета.
"""

import httpx
import streamlit as st

from streamlit_app.components.prediction_card import render_prediction_card
from streamlit_app.components.similar_players import render_similar_players
from streamlit_app.components.shap_chart import render_shap_chart

API_URL = "http://api:8000"


def _search(name: str, player_type: str) -> list[dict]:
    try:
        r = httpx.get(
            f"{API_URL}/players/search",
            params={"name": name, "type": player_type},
            timeout=10,
        )
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ошибка поиска: {e}")
        return []


def _predict(payload: dict, player_type: str) -> dict | None:
    try:
        r = httpx.post(
            f"{API_URL}/predict/{player_type}",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")
        return None


def _stats_from_player(player: dict, player_type: str) -> dict:
    """Строим payload для API из найденного игрока."""
    stats = player["stats"]
    base = {
        "season_year": player["season_year"],
        "gamesPlayed": stats.get("gamesPlayed", 60),
        "heightInInches": 73,
        "weightInPounds": 200,
    }
    if player_type == "skater":
        base.update({
            "positionCode": player["position"],
            "goals": stats.get("goals", 0),
            "assists": stats.get("assists", 0),
            "points": stats.get("points", 0),
            "plusMinus": stats.get("plusMinus", 0),
            "timeOnIcePerGame": stats.get("timeOnIcePerGame", 900),
        })
    else:
        base.update({
            "wins": stats.get("wins", 0),
            "savePct": stats.get("savePct", 0.910),
            "goalsAgainstAverage": stats.get("goalsAgainstAverage", 2.8),
            "gamesStarted": stats.get("gamesStarted", 30),
        })
    return base


def _render_player_card(player: dict, player_type: str):
    """Карточка найденного игрока с его реальной статистикой."""
    stats = player["stats"]
    pos = player["position"]
    team = player["team"]
    season = player["season_year"]

    if player_type == "skater":
        stat_lines = [
            f"**Голы:** {stats.get('goals', '—')}",
            f"**Передачи:** {stats.get('assists', '—')}",
            f"**Очки:** {stats.get('points', '—')}",
            f"**+/-:** {stats.get('plusMinus', '—')}",
            f"**TOI/игру:** {stats.get('timeOnIcePerGame', 0)/60:.1f} мин",
        ]
    else:
        stat_lines = [
            f"**Победы:** {stats.get('wins', '—')}",
            f"**SV%:** {stats.get('savePct', 0):.3f}",
            f"**GAA:** {stats.get('goalsAgainstAverage', '—')}",
            f"**Стартов:** {stats.get('gamesStarted', '—')}",
        ]

    st.markdown(
        f"""
        <div style="
            background: #1F2937;
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
        ">
            <div style="font-size: 20px; font-weight: 800; color: #F9FAFB;">
                {player['name']}
            </div>
            <div style="font-size: 13px; color: #9CA3AF; margin-top: 2px;">
                {pos} · {team} · Сезон {season-1}–{str(season)[-2:]}
            </div>
            <div style="
                font-size: 22px;
                font-weight: 700;
                color: #34D399;
                margin-top: 10px;
            ">
                Реальный cap hit: {player['cap_hit_m']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(len(stat_lines))
    for col, line in zip(cols, stat_lines):
        with col:
            st.markdown(line)


def render(player_type: str):
    st.markdown("### 🔍 Поиск реального игрока")
    st.caption("Найди игрока из датасета — модель предскажет его зарплату и сравнит с реальной")

    name_query = st.text_input(
        "Введи фамилию или имя",
        placeholder="Например: Pettersson, McDavid, Vasilevskiy...",
    )

    if not name_query or len(name_query) < 2:
        st.info("Введи минимум 2 символа для поиска")
        return

    results = _search(name_query, player_type)

    if not results:
        st.warning(f"Игроки по запросу «{name_query}» не найдены в датасете (сезоны 2022–2026)")
        return

    if len(results) == 1:
        selected = results[0]
    else:
        options = {
            f"{r['name']} · {r['position']} · {r['team']} · {r['season_year']}": r
            for r in results
        }
        choice = st.selectbox("Найдено несколько игроков — выбери нужного:", list(options.keys()))
        selected = options[choice]

    st.divider()
    _render_player_card(selected, player_type)

    payload = _stats_from_player(selected, player_type)

    with st.spinner("Предсказываем зарплату..."):
        result = _predict(payload, player_type)

    if not result:
        return

    actual = selected["cap_hit"]

    render_prediction_card(result, actual_cap_hit=actual)

    col1, col2 = st.columns([1, 1])
    with col1:
        render_shap_chart(result["shap_top5"])
    with col2:
        render_similar_players(result["similar_players"])

    st.divider()
    st.markdown("#### 📊 Детали предсказания")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Реальный cap hit", f"${actual/1e6:.2f}M")
    c2.metric("Прогноз модели", result["predicted_cap_hit_m"])
    pred = result["predicted_cap_hit"]
    err = pred - actual
    err_pct = err / actual * 100
    c3.metric("Абсолютная ошибка", f"${abs(err)/1e6:.2f}M")
    c4.metric("Относительная ошибка", f"{err_pct:+.1f}%",
              delta_color="inverse" if abs(err_pct) > 20 else "off")