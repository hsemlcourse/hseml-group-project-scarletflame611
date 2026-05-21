"""
SHAP waterfall барчарт для топ-5 признаков.
"""

import plotly.graph_objects as go
import streamlit as st

FEATURE_LABELS = {
    "timeOnIcePerGame":        "Время на льду / игру",
    "assists":                 "Передачи",
    "age":                     "Возраст",
    "age_x_points_per_60":     "Возраст × продуктивность",
    "ppTimeOnIcePerGame":      "Время в большинстве",
    "draft_value":             "Ценность на драфте",
    "hits_per_60_x_defense":   "Хиты (защитники)",
    "points_per_60":           "Очки за 60 мин",
    "goals_per_60":            "Голы за 60 мин",
    "assists_per_60":          "Передачи за 60 мин",
    "goals":                   "Голы",
    "points":                  "Очки",
    "plusMinus":               "+/-",
    "penaltyMinutes":          "Штрафные минуты",
    "weightInPounds":          "Вес",
    "size_index":              "Физические данные",
    "qualityStart":            "Качественные старты",
    "gamesStarted":            "Стартов",
    "starter_ratio":           "Доля стартов",
    "gsaa_proxy":              "GSAA (голы сэкономлены)",
    "savePct":                 "SV%",
    "win_rate":                "% побед",
}


def render_shap_chart(shap_top5: list[dict]):
    if not shap_top5:
        return

    st.markdown("#### 🔍 Почему такая зарплата")
    st.caption("Вклад каждого признака в предсказание относительно среднего по датасету")

    features = [FEATURE_LABELS.get(f["feature"], f["feature"]) for f in shap_top5]
    shap_vals = [f["shap_value"] for f in shap_top5]
    colors = ["#34D399" if v > 0 else "#F87171" for v in shap_vals]
    text_labels = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals,
        y=features,
        orientation="h",
        marker_color=colors,
        text=text_labels,
        textposition="outside",
        textfont=dict(size=12, color="#D1D5DB"),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="#374151",
            zeroline=True,
            zerolinecolor="#6B7280",
            zerolinewidth=1.5,
            tickfont=dict(color="#9CA3AF"),
            title=dict(text="SHAP value (влияние на log_cap_hit)", font=dict(color="#9CA3AF")),
        ),
        yaxis=dict(
            tickfont=dict(color="#D1D5DB", size=12),
            autorange="reversed",
        ),
        margin=dict(l=10, r=60, t=10, b=40),
        height=260,
    )

    st.plotly_chart(fig, use_container_width=True)