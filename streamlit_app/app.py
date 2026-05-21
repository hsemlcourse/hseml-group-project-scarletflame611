"""
Точка входа Streamlit. Навигация и глобальный переключатель скейтер/вратарь.
"""

import httpx
import streamlit as st

from streamlit_app.pages import form, search, batch

API_URL = "http://api:8000"

st.set_page_config(
    page_title="NHL Salary Predictor",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 52px;
        letter-spacing: 3px;
        color: #F9FAFB;
        line-height: 1;
        margin-bottom: 4px;
    }

    .main-subtitle {
        font-size: 15px;
        color: #6B7280;
        margin-bottom: 0;
    }

    div[data-testid="stHorizontalBlock"] button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }

    .stButton > button[kind="primary"] {
        background: #10B981;
        border: none;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .stButton > button[kind="primary"]:hover {
        background: #059669;
    }

    hr {
        border-color: #374151;
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)


def _check_api() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            '<div class="main-title">🏒 NHL SALARY PREDICTOR</div>'
            '<div class="main-subtitle">'
            'Предсказание cap hit хоккеистов по игровой статистике · '
            'Сезоны 2021–22 → 2025–26'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        api_ok = _check_api()
        if api_ok:
            st.markdown(
                '<div style="text-align:right; padding-top: 16px;">'
                '<span style="color:#10B981; font-size:13px; font-weight:600;">'
                '● API online</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:right; padding-top: 16px;">'
                '<span style="color:#EF4444; font-size:13px; font-weight:600;">'
                '● API offline</span></div>',
                unsafe_allow_html=True,
            )
            st.warning("API недоступен. Убедись что `docker-compose up` запущен.")


def _render_model_metrics():
    try:
        r = httpx.get(f"{API_URL}/model-info", timeout=5)
        if r.status_code != 200:
            return
        info = r.json()
    except Exception:
        return

    with st.expander("📊 Метрики финальных моделей", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Скейтеры — MAE", "$1.247M")
        c2.metric("Скейтеры — MAPE", "31.5%")
        c3.metric("Скейтеры — R²", "0.667")
        c4.metric("Вратари — MAE", "$1.251M")
        c5.metric("Вратари — R²", "0.529")
        st.caption(
            f"Скейтеры: {info['skaters']['model']} · "
            f"{info['skaters']['n_features']} признаков · "
            f"Вратари: {info['goalies']['model']} · "
            f"Данные: {info['seasons']}"
        )


def _render_nav() -> tuple[str, str]:
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    col_type, col_nav = st.columns([1, 3])

    with col_type:
        player_type = st.radio(
            "Тип игрока",
            options=["skater", "goalie"],
            format_func=lambda x: "🏒 Скейтер" if x == "skater" else "🥅 Вратарь",
            horizontal=True,
            key="player_type",
        )

    with col_nav:
        tabs = ["✏️ Ввод вручную", "🔍 Поиск игрока", "📂 Пакетное предсказание"]
        if "active_tab" not in st.session_state:
            st.session_state["active_tab"] = tabs[0]

        c1, c2, c3, _ = st.columns([1, 1, 1, 2])
        with c1:
            if st.button(tabs[0], use_container_width=True,
                         type="primary" if st.session_state["active_tab"] == tabs[0] else "secondary"):
                st.session_state["active_tab"] = tabs[0]
        with c2:
            if st.button(tabs[1], use_container_width=True,
                         type="primary" if st.session_state["active_tab"] == tabs[1] else "secondary"):
                st.session_state["active_tab"] = tabs[1]
        with c3:
            if st.button(tabs[2], use_container_width=True,
                         type="primary" if st.session_state["active_tab"] == tabs[2] else "secondary"):
                st.session_state["active_tab"] = tabs[2]

    return player_type, st.session_state["active_tab"]


def main():
    _render_header()
    st.divider()
    _render_model_metrics()

    player_type, active_tab = _render_nav()
    st.divider()

    if active_tab == "✏️ Ввод вручную":
        form.render(player_type)
    elif active_tab == "🔍 Поиск игрока":
        search.render(player_type)
    elif active_tab == "📂 Пакетное предсказание":
        batch.render(player_type)


if __name__ == "__main__":
    main()