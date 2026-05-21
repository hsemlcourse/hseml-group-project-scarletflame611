"""
Карточки похожих игроков по рыночной стоимости.
"""

import streamlit as st


def render_similar_players(players: list[dict]):
    if not players:
        return

    st.markdown("#### 🏒 Игроки с похожей рыночной ценностью")
    st.caption("Реальные игроки из датасета с близким cap hit")

    cols = st.columns(len(players))
    for col, player in zip(cols, players):
        with col:
            st.markdown(
                f"""
                <div style="
                    background: #1F2937;
                    border: 1px solid #374151;
                    border-radius: 12px;
                    padding: 14px;
                    text-align: center;
                    height: 100%;
                ">
                    <div style="font-size: 13px; font-weight: 700; color: #F9FAFB; line-height: 1.3;">
                        {player['name']}
                    </div>
                    <div style="font-size: 11px; color: #9CA3AF; margin-top: 4px;">
                        {player['position']} · {player['team']}
                    </div>
                    <div style="
                        font-size: 18px;
                        font-weight: 800;
                        color: #34D399;
                        margin-top: 8px;
                    ">
                        {player['cap_hit_m']}
                    </div>
                    <div style="font-size: 11px; color: #6B7280; margin-top: 6px;">
                        {player['key_stat']}: <span style="color: #D1D5DB;">{player['key_stat_value']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )