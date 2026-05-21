"""
Режим пакетного предсказания: загрузка CSV и предсказание для списка игроков.
"""

import io

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = "http://api:8000"

SKATER_TEMPLATE_COLS = [
    "skaterFullName", "gamesPlayed", "goals", "assists", "points",
    "plusMinus", "penaltyMinutes", "shots", "timeOnIcePerGame",
    "ppGoals", "ppPoints", "ppTimeOnIcePerGame",
    "hits", "blockedShots", "heightInInches", "weightInPounds",
    "birthCountry", "draftYear",
]

GOALIE_TEMPLATE_COLS = [
    "goalieFullName", "gamesPlayed", "gamesStarted", "wins",
    "savePct", "goalsAgainstAverage", "shotsAgainst",
    "shutouts", "qualityStart",
    "heightInInches", "weightInPounds", "birthCountry", "draftYear",
]

SKATER_DEFAULTS = {
    "skaterFullName": "John Doe",
    "gamesPlayed": 60,
    "goals": 15,
    "assists": 25,
    "points": 40,
    "plusMinus": 0,
    "penaltyMinutes": 20,
    "shots": 120,
    "timeOnIcePerGame": 900,
    "ppGoals": 3,
    "ppPoints": 8,
    "ppTimeOnIcePerGame": 90,
    "hits": 50,
    "blockedShots": 30,
    "heightInInches": 73,
    "weightInPounds": 200,
    "birthCountry": "CAN",
    "draftYear": 2018,
}

GOALIE_DEFAULTS = {
    "goalieFullName": "John Goalie",
    "gamesPlayed": 40,
    "gamesStarted": 35,
    "wins": 18,
    "savePct": 0.910,
    "goalsAgainstAverage": 2.8,
    "shotsAgainst": 1100,
    "shutouts": 2,
    "qualityStart": 18,
    "heightInInches": 74,
    "weightInPounds": 195,
    "birthCountry": "CAN",
    "draftYear": 2016,
}

TIER_COLORS = {
    "Entry-level": "#6B7280",
    "Bottom-6":    "#3B82F6",
    "Middle-6":    "#10B981",
    "Top-6":       "#F59E0B",
    "Star":        "#EF4444",
    "Franchise":   "#8B5CF6",
}


def _make_template(player_type: str) -> bytes:
    if player_type == "skater":
        df = pd.DataFrame([SKATER_DEFAULTS])
    else:
        df = pd.DataFrame([GOALIE_DEFAULTS])
    return df.to_csv(index=False).encode("utf-8")


def _call_batch_api(file_bytes: bytes, filename: str) -> dict | None:
    try:
        r = httpx.post(
            f"{API_URL}/predict/batch",
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Ошибка API: {e.response.text}")
    except Exception as e:
        st.error(f"Не удалось подключиться к API: {e}")
    return None


def _render_tier_chart(tier_dist: dict[str, int]):
    if not tier_dist:
        return

    tiers_ordered = [
        "Entry-level", "Bottom-6", "Middle-6",
        "Top-6", "Star", "Franchise",
    ]
    labels = [t for t in tiers_ordered if t in tier_dist]
    values = [tier_dist[t] for t in labels]
    colors = [TIER_COLORS[t] for t in labels]

    fig = px.bar(
        x=labels,
        y=values,
        color=labels,
        color_discrete_map=TIER_COLORS,
        labels={"x": "Salary Tier", "y": "Игроков"},
        title="Распределение по salary tier",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(tickfont=dict(color="#D1D5DB")),
        yaxis=dict(
            tickfont=dict(color="#9CA3AF"),
            gridcolor="#374151",
        ),
        title_font=dict(color="#F9FAFB"),
        margin=dict(t=40, b=20),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_salary_distribution(predictions: list[dict]):
    values = [p["predicted_cap_hit"] / 1e6 for p in predictions
              if p["predicted_cap_hit"] > 0]
    if not values:
        return

    fig = px.histogram(
        x=values,
        nbins=20,
        labels={"x": "Предсказанный cap hit, $M", "y": "Игроков"},
        title="Распределение предсказанных зарплат",
        color_discrete_sequence=["#34D399"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#D1D5DB"), gridcolor="#374151"),
        yaxis=dict(tickfont=dict(color="#9CA3AF"), gridcolor="#374151"),
        title_font=dict(color="#F9FAFB"),
        margin=dict(t=40, b=20),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_top5(predictions: list[dict]):
    valid = [p for p in predictions if p["predicted_cap_hit"] > 0]
    top5 = sorted(valid, key=lambda x: x["predicted_cap_hit"], reverse=True)[:5]

    st.markdown("#### 🏆 Топ-5 по предсказанной зарплате")
    for i, p in enumerate(top5, 1):
        tier_color = TIER_COLORS.get(p["salary_tier"], "#6B7280")
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 16px;
                background: #1F2937;
                border: 1px solid #374151;
                border-radius: 10px;
                padding: 12px 18px;
                margin-bottom: 8px;
            ">
                <div style="font-size: 20px; font-weight: 800; color: #4B5563; width: 28px;">
                    {i}
                </div>
                <div style="flex: 1;">
                    <div style="font-weight: 700; color: #F9FAFB;">{p['name']}</div>
                    <div style="font-size: 12px; color: #6B7280; margin-top: 2px;">
                        <span style="
                            color: {tier_color};
                            background: {tier_color}22;
                            padding: 2px 8px;
                            border-radius: 10px;
                            font-size: 11px;
                        ">{p['salary_tier']}</span>
                    </div>
                </div>
                <div style="font-size: 20px; font-weight: 800; color: #34D399;">
                    {p['predicted_cap_hit_m']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render(player_type: str):
    st.markdown("### 📂 Пакетное предсказание")
    st.caption("Загрузи CSV с игроками — получи предсказания для всех сразу")

    col1, col2 = st.columns([3, 1])
    with col2:
        template_bytes = _make_template(player_type)
        st.download_button(
            label="⬇️ Скачать шаблон CSV",
            data=template_bytes,
            file_name=f"nhl_{player_type}s_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col1:
        uploaded = st.file_uploader(
            "Загрузи CSV файл",
            type=["csv"],
            help="Используй шаблон справа чтобы убедиться в правильном формате",
        )

    if uploaded is None:
        st.info("Загрузи CSV чтобы начать. Скачай шаблон справа если нужен пример формата.")
        return

    file_bytes = uploaded.read()

    try:
        preview_df = pd.read_csv(io.BytesIO(file_bytes))
        st.markdown(f"**Загружено:** {len(preview_df)} игроков, {len(preview_df.columns)} колонок")
        with st.expander("Предпросмотр данных"):
            st.dataframe(preview_df.head(5), use_container_width=True)
    except Exception:
        st.error("Не удалось прочитать файл. Проверь формат CSV.")
        return

    if st.button("🚀 Предсказать для всех", type="primary", use_container_width=True):
        with st.spinner(f"Обрабатываем {len(preview_df)} игроков..."):
            result = _call_batch_api(file_bytes, uploaded.name)

        if not result:
            return

        st.success(f"Готово! Обработано {result['count']} игроков")
        st.divider()

        predictions = result["predictions"]
        tier_dist = result["tier_distribution"]

        col1, col2 = st.columns(2)
        with col1:
            _render_tier_chart(tier_dist)
        with col2:
            _render_salary_distribution(predictions)

        st.divider()
        _render_top5(predictions)

        st.divider()
        st.markdown("#### 📋 Полная таблица результатов")

        result_df = pd.DataFrame(predictions)
        result_df = result_df.rename(columns={
            "name": "Игрок",
            "predicted_cap_hit_m": "Прогноз",
            "salary_tier": "Tier",
            "predicted_cap_hit": "cap_hit_usd",
        })
        result_df = result_df.sort_values("cap_hit_usd", ascending=False)
        st.dataframe(
            result_df[["Игрок", "Прогноз", "Tier"]],
            use_container_width=True,
            hide_index=True,
        )

        csv_out = result_df[["Игрок", "Прогноз", "Tier", "cap_hit_usd"]].to_csv(
            index=False
        ).encode("utf-8")
        st.download_button(
            label="⬇️ Скачать результаты",
            data=csv_out,
            file_name="nhl_predictions.csv",
            mime="text/csv",
        )