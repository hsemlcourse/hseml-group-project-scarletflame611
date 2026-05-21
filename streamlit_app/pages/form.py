"""
Режим ручного ввода статистики с слайдерами "что если".
"""

import httpx
import streamlit as st

from streamlit_app.components.prediction_card import render_prediction_card
from streamlit_app.components.similar_players import render_similar_players
from streamlit_app.components.shap_chart import render_shap_chart

API_URL = "http://api:8000"


def _predict(payload: dict, player_type: str) -> dict | None:
    endpoint = f"{API_URL}/predict/{player_type}"
    try:
        r = httpx.post(endpoint, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Ошибка API: {e.response.text}")
    except Exception as e:
        st.error(f"Не удалось подключиться к API: {e}")
    return None


def render_skater_form() -> dict:
    st.markdown("##### Основное")
    c1, c2, c3 = st.columns(3)
    with c1:
        position = st.selectbox("Позиция", ["C", "L", "R", "D"])
    with c2:
        games = st.number_input("Матчей сыграно", 1, 82, 60)
    with c3:
        toi = st.slider("Время на льду / игру (мин)", 5.0, 30.0, 15.0, 0.5)

    st.markdown("##### Результативность")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        goals = st.number_input("Голы", 0, 100, 15)
    with c2:
        assists = st.number_input("Передачи", 0, 100, 25)
    with c3:
        points = st.number_input("Очки", 0, 200, goals + assists)
    with c4:
        plus_minus = st.number_input("+/-", -50, 50, 0)

    st.markdown("##### Большинство")
    c1, c2, c3 = st.columns(3)
    with c1:
        pp_points = st.number_input("Очки в большинстве", 0, 60, 8)
    with c2:
        pp_goals = st.number_input("Голы в большинстве", 0, 30, 3)
    with c3:
        pp_toi = st.slider("Время в большинстве / игру (мин)", 0.0, 8.0, 1.5, 0.1)

    st.markdown("##### Физические действия")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hits = st.number_input("Хиты", 0, 400, 50)
    with c2:
        blocks = st.number_input("Заблокированные броски", 0, 300, 30)
    with c3:
        shots = st.number_input("Броски", 0, 400, 120)
    with c4:
        penalty = st.number_input("Штрафные минуты", 0, 200, 20)

    st.markdown("##### Биография")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        height = st.number_input("Рост (дюймы)", 60, 84, 73)
    with c2:
        weight = st.number_input("Вес (фунты)", 150, 280, 200)
    with c3:
        country = st.selectbox("Страна", ["CAN", "USA", "RUS", "SWE", "FIN", "CZE", "OTHER"])
    with c4:
        draft_year = st.number_input("Год драфта", 1990, 2025, 2018, step=1)

    with st.expander("📋 Статистика прошлого сезона (для точности)"):
        st.caption("Если заполнить — модель учтёт динамику развития игрока")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            prev_goals = st.number_input("Голы (прошлый сезон)", 0, 100, 0, key="prev_g")
            prev_assists = st.number_input("Передачи (прошлый сезон)", 0, 100, 0, key="prev_a")
        with pc2:
            prev_points = st.number_input("Очки (прошлый сезон)", 0, 200, 0, key="prev_p")
            prev_toi = st.slider("Время на льду (прошлый сезон, мин)", 0.0, 30.0, 0.0, key="prev_toi")
        with pc3:
            prev_hits = st.number_input("Хиты (прошлый сезон)", 0, 400, 0, key="prev_h")
            prev_cap = st.number_input("Зарплата (прошлый сезон, $M)", 0.0, 20.0, 0.0, key="prev_cap")

    prev_season = None
    if any([prev_goals, prev_assists, prev_points, prev_toi, prev_hits, prev_cap]):
        prev_season = {
            "goals": prev_goals,
            "assists": prev_assists,
            "points": prev_points,
            "timeOnIcePerGame": prev_toi * 60,
            "hits": prev_hits,
            "cap_hit": prev_cap * 1_000_000,
        }

    return {
        "positionCode": position,
        "season_year": 2026,
        "gamesPlayed": games,
        "goals": goals,
        "assists": assists,
        "points": points,
        "plusMinus": plus_minus,
        "penaltyMinutes": penalty,
        "shots": shots,
        "timeOnIcePerGame": toi * 60,
        "ppGoals": pp_goals,
        "ppPoints": pp_points,
        "ppAssists": pp_points - pp_goals,
        "ppTimeOnIcePerGame": pp_toi * 60,
        "hits": hits,
        "blockedShots": blocks,
        "heightInInches": height,
        "weightInPounds": weight,
        "birthCountry": country,
        "draftYear": draft_year,
        "prev_season": prev_season,
    }


def render_goalie_form() -> dict:
    st.markdown("##### Основное")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        games = st.number_input("Матчей сыграно", 1, 82, 40)
    with c2:
        started = st.number_input("Стартов", 0, 82, 35)
    with c3:
        wins = st.number_input("Победы", 0, 60, 18)
    with c4:
        shutouts = st.number_input("Игры на ноль", 0, 20, 2)

    st.markdown("##### Качество игры")
    c1, c2, c3 = st.columns(3)
    with c1:
        sv_pct = st.slider("SV% (процент отражённых)", 0.870, 0.950, 0.910, 0.001,
                           format="%.3f")
    with c2:
        gaa = st.slider("GAA (пропущено / игру)", 1.5, 4.5, 2.8, 0.1)
    with c3:
        quality_starts = st.number_input("Качественные старты", 0, 60, 18)

    st.markdown("##### Биография")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        height = st.number_input("Рост (дюймы)", 60, 84, 74)
    with c2:
        weight = st.number_input("Вес (фунты)", 150, 280, 195)
    with c3:
        country = st.selectbox("Страна", ["CAN", "USA", "RUS", "SWE", "FIN", "CZE", "OTHER"])
    with c4:
        draft_year = st.number_input("Год драфта", 1990, 2025, 2018, step=1)

    shots_against = int(games * gaa / (1 - sv_pct)) if sv_pct < 1 else 0

    with st.expander("📋 Статистика прошлого сезона (для точности)"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            prev_sv = st.slider("SV% (прошлый сезон)", 0.870, 0.950, 0.910, 0.001, key="prev_sv")
            prev_wins = st.number_input("Победы (прошлый сезон)", 0, 60, 0, key="prev_w")
        with pc2:
            prev_gaa = st.slider("GAA (прошлый сезон)", 1.5, 4.5, 2.8, 0.1, key="prev_gaa")
            prev_started = st.number_input("Стартов (прошлый сезон)", 0, 82, 0, key="prev_st")
        with pc3:
            prev_cap = st.number_input("Зарплата (прошлый сезон, $M)", 0.0, 20.0, 0.0, key="prev_cap_g")

    prev_season = None
    if any([prev_wins, prev_started, prev_cap]):
        prev_season = {
            "savePct": prev_sv,
            "goalsAgainstAverage": prev_gaa,
            "wins": prev_wins,
            "gamesStarted": prev_started,
            "cap_hit": prev_cap * 1_000_000,
        }

    return {
        "season_year": 2026,
        "gamesPlayed": games,
        "gamesStarted": started,
        "wins": wins,
        "shutouts": shutouts,
        "savePct": sv_pct,
        "goalsAgainstAverage": gaa,
        "qualityStart": quality_starts,
        "shotsAgainst": shots_against,
        "heightInInches": height,
        "weightInPounds": weight,
        "birthCountry": country,
        "draftYear": draft_year,
        "prev_season": prev_season,
    }


def _render_whatif_sliders(result: dict, base_payload: dict, player_type: str):
    st.markdown("#### 🎛️ Что если...")
    st.caption("Подвигай слайдеры — зарплата пересчитается автоматически")

    top_features = [f["feature"] for f in result["shap_top5"][:3]]

    WHAT_IF_CONFIG = {
        "timeOnIcePerGame": {
            "label": "Время на льду / игру (мин)",
            "min": 5.0, "max": 30.0, "step": 0.5,
            "to_payload": lambda v: v * 60,
            "from_payload": lambda p: p.get("timeOnIcePerGame", 900) / 60,
        },
        "assists": {
            "label": "Передачи",
            "min": 0, "max": 80, "step": 1,
            "to_payload": lambda v: int(v),
            "from_payload": lambda p: p.get("assists", 0),
        },
        "goals": {
            "label": "Голы",
            "min": 0, "max": 60, "step": 1,
            "to_payload": lambda v: int(v),
            "from_payload": lambda p: p.get("goals", 0),
        },
        "ppTimeOnIcePerGame": {
            "label": "Время в большинстве / игру (мин)",
            "min": 0.0, "max": 8.0, "step": 0.1,
            "to_payload": lambda v: v * 60,
            "from_payload": lambda p: p.get("ppTimeOnIcePerGame", 0) / 60,
        },
        "hits": {
            "label": "Хиты",
            "min": 0, "max": 300, "step": 5,
            "to_payload": lambda v: int(v),
            "from_payload": lambda p: p.get("hits", 0),
        },
        "age": {
            "label": "Возраст",
            "min": 18, "max": 42, "step": 1,
            "to_payload": lambda v: None,
            "from_payload": lambda p: 26,
        },
        "savePct": {
            "label": "SV%",
            "min": 0.870, "max": 0.950, "step": 0.001,
            "to_payload": lambda v: float(v),
            "from_payload": lambda p: p.get("savePct", 0.910),
        },
        "wins": {
            "label": "Победы",
            "min": 0, "max": 50, "step": 1,
            "to_payload": lambda v: int(v),
            "from_payload": lambda p: p.get("wins", 0),
        },
    }

    relevant = [f for f in top_features if f in WHAT_IF_CONFIG]
    if not relevant:
        fallback = ["timeOnIcePerGame", "assists", "goals"] if player_type == "skater" \
            else ["savePct", "wins", "gamesStarted"]
        relevant = [f for f in fallback if f in WHAT_IF_CONFIG][:3]

    wi_payload = base_payload.copy()
    changed = False

    cols = st.columns(len(relevant))
    for col, feat in zip(cols, relevant):
        cfg = WHAT_IF_CONFIG[feat]
        base_val = cfg["from_payload"](base_payload)
        with col:
            new_val = st.slider(
                cfg["label"],
                cfg["min"], cfg["max"],
                float(base_val),
                cfg["step"],
                key=f"wi_{feat}",
            )
            if new_val != base_val:
                converted = cfg["to_payload"](new_val)
                if converted is not None:
                    wi_payload[feat] = converted
                    changed = True

    if changed:
        wi_result = _predict(wi_payload, player_type)
        if wi_result:
            base_pred = result["predicted_cap_hit"]
            new_pred = wi_result["predicted_cap_hit"]
            delta = new_pred - base_pred
            delta_pct = delta / base_pred * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Базовый прогноз", f"${base_pred/1e6:.2f}M")
            c2.metric("Новый прогноз", f"${new_pred/1e6:.2f}M",
                      delta=f"{delta_pct:+.1f}%",
                      delta_color="normal")
            c3.metric("Изменение", f"${delta/1e6:+.2f}M")


def render(player_type: str):
    st.markdown("### ✏️ Ввод статистики вручную")

    if player_type == "skater":
        payload = render_skater_form()
    else:
        payload = render_goalie_form()

    if st.button("🏒 Предсказать зарплату", type="primary", use_container_width=True):
        with st.spinner("Считаем..."):
            result = _predict(payload, player_type)

        if result:
            st.session_state["last_result"] = result
            st.session_state["last_payload"] = payload
            st.session_state["last_player_type"] = player_type

    if "last_result" in st.session_state and \
            st.session_state.get("last_player_type") == player_type:

        result = st.session_state["last_result"]
        payload = st.session_state["last_payload"]

        st.divider()
        render_prediction_card(result)

        col1, col2 = st.columns([1, 1])
        with col1:
            render_shap_chart(result["shap_top5"])
        with col2:
            render_similar_players(result["similar_players"])

        st.divider()
        _render_whatif_sliders(result, payload, player_type)