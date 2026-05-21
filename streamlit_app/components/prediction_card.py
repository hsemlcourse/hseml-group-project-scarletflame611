"""
Карточка с результатом предсказания: зарплата, tier, бейдж справедливости.
"""

import streamlit as st

TIER_COLORS = {
    "Entry-level": "#6B7280",
    "Bottom-6":    "#3B82F6",
    "Middle-6":    "#10B981",
    "Top-6":       "#F59E0B",
    "Star":        "#EF4444",
    "Franchise":   "#8B5CF6",
}

TIER_EMOJI = {
    "Entry-level": "🔰",
    "Bottom-6":    "🏒",
    "Middle-6":    "⭐",
    "Top-6":       "🌟",
    "Star":        "💫",
    "Franchise":   "👑",
}


def render_prediction_card(result: dict, actual_cap_hit: float | None = None):
    """
    result: словарь из PredictionOutput
    actual_cap_hit: реальная зарплата — передаётся только в режиме поиска игрока
    """
    predicted = result["predicted_cap_hit"]
    tier = result["salary_tier"]
    tier_range = result["tier_range"]
    color = TIER_COLORS.get(tier, "#6B7280")
    emoji = TIER_EMOJI.get(tier, "🏒")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}18, {color}08);
            border: 2px solid {color}40;
            border-radius: 16px;
            padding: 28px 32px;
            text-align: center;
            margin-bottom: 16px;
        ">
            <div style="font-size: 14px; color: #9CA3AF; margin-bottom: 4px; letter-spacing: 1px; text-transform: uppercase;">
                Предсказанный cap hit
            </div>
            <div style="font-size: 52px; font-weight: 800; color: {color}; line-height: 1.1;">
                {result["predicted_cap_hit_m"]}
            </div>
            <div style="margin-top: 12px;">
                <span style="
                    background: {color}22;
                    border: 1px solid {color}60;
                    border-radius: 20px;
                    padding: 4px 16px;
                    font-size: 14px;
                    font-weight: 600;
                    color: {color};
                ">
                    {emoji} {tier} · {tier_range}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if actual_cap_hit is not None:
        _render_fairness_badge(predicted, actual_cap_hit)


def _render_fairness_badge(predicted: float, actual: float):
    diff_pct = (predicted - actual) / actual * 100

    if diff_pct > 20:
        label = "Переоценён рынком"
        description = f"Модель оценивает на {abs(diff_pct):.0f}% выше реального контракта"
        bg = "#FEF3C7"
        border = "#F59E0B"
        text = "#92400E"
        icon = "📈"
    elif diff_pct < -20:
        label = "Недооценён рынком"
        description = f"Модель оценивает на {abs(diff_pct):.0f}% ниже реального контракта"
        bg = "#EFF6FF"
        border = "#3B82F6"
        text = "#1E40AF"
        icon = "📉"
    else:
        label = "Рыночная цена"
        description = f"Отклонение {diff_pct:+.0f}% — модель попала в диапазон"
        bg = "#F0FDF4"
        border = "#10B981"
        text = "#065F46"
        icon = "✅"

    actual_m = f"${actual/1e6:.2f}M"
    predicted_m = f"${predicted/1e6:.2f}M"

    st.markdown(
        f"""
        <div style="
            background: {bg};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 16px;
        ">
            <div style="font-weight: 700; color: {text}; font-size: 15px;">
                {icon} {label}
            </div>
            <div style="color: {text}; font-size: 13px; margin-top: 4px; opacity: 0.85;">
                {description}
            </div>
            <div style="
                display: flex;
                gap: 24px;
                margin-top: 12px;
                font-size: 13px;
                color: {text};
            ">
                <span>Реально: <b>{actual_m}</b></span>
                <span>Прогноз: <b>{predicted_m}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )