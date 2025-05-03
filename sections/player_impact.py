import streamlit as st
import pandas as pd

def show_impact(points_df: pd.DataFrame, available_matches_df: pd.DataFrame, n_matches_played: int):
    st.subheader("🧠 Player Impact - Next Match Focus", divider="orange")

    available_matches = available_matches_df["Match"].tolist()

    match_input = st.selectbox("Current/Next Match", options=available_matches, index=0)
    st.session_state.match_input = match_input

    if match_input:
        teams = [t.strip() for t in match_input.split("vs")]
        if len(teams) == 2:
            exclusion_pattern = r"\(O\)|\(RE\)"
            valid_players_df = points_df[
                (points_df["Team"].isin(teams)) &
                (~points_df["Player Name"].str.contains(exclusion_pattern, regex=True))
            ].copy()

            if "Player Name" in valid_players_df.columns:
                non_playing_options = sorted(valid_players_df["Player Name"].unique())
            else:
                non_playing_options = []
                st.warning("No players found for the selected match.")

            non_playing = st.multiselect(
                f"Select Non-Playing Players from {teams[0]}/{teams[1]}:",
                options=non_playing_options,
                key=f"non_playing_{match_input.replace(' ', '_')}" # Unique key per match
            )

            # Set playing status (still needed for filtering)
            valid_players_df["Playing Status"] = valid_players_df["Player Name"].apply(
                lambda x: "❌ Not Playing" if x in non_playing else "✅ Playing"
            )

            # Calculate projected points
            valid_players_df["Projected Points"] = (valid_players_df["Total Points"] / n_matches_played).round(1)

            # Style category for Projected Points
            def point_color_indicator(x):
                if x > 30:
                    return "🟢 High"
                elif x > 20:
                    return "🟠 Medium"
                else:
                    return "⚪ Low"

            valid_players_df["Impact"] = valid_players_df["Projected Points"].apply(point_color_indicator)

            # Filter out non-playing players for display
            playing_players_df = valid_players_df[valid_players_df["Playing Status"] == "✅ Playing"].copy()

            # Final display columns with renamed columns
            display_df = playing_players_df[[
                "Owner", "Impact", "Player Name", "Team", "Projected Points"
            ]].sort_values(by="Owner").rename(columns={
                "Owner": "Team Owner",
                "Impact": "Impact",
                "Player Name": "Player",
                "Team": "Team",
                "Projected Points": "Projection"
            })

            st.markdown("### <span style='font-size: 0.8em;'>📊 Player Projections for the Upcoming Match</span>", unsafe_allow_html=True)

            unique_owners = sorted(display_df["Team Owner"].unique())
            num_owners = len(unique_owners)

            if num_owners > 0:
                if num_owners <= 5:
                    for owner in unique_owners:
                        owner_df = display_df[display_df["Team Owner"] == owner].drop(columns=["Team Owner"])
                        st.markdown(f"#### <span style='font-size: 1em;'>Team Owner: {owner}</span>", unsafe_allow_html=True)
                        st.dataframe(owner_df.style.set_properties(**{'font-size': '0.7em'}).format(subset=['Projection'], formatter="{:.1f}"), use_container_width=True, hide_index=True)
                        st.markdown("<hr style='margin: 0.8rem 0;'/>", unsafe_allow_html=True)
                else:
                    col1, col2 = st.columns(2)
                    owner_groups = [unique_owners[:num_owners//2], unique_owners[num_owners//2:]]

                    with col1:
                        for owner in owner_groups[0]:
                            owner_df = display_df[display_df["Team Owner"] == owner].drop(columns=["Team Owner"])
                            st.markdown(f"#### <span style='font-size: 1em;'>Team Owner: {owner}</span>", unsafe_allow_html=True)
                            st.dataframe(owner_df.style.set_properties(**{'font-size': '0.7em'}).format(subset=['Projection'], formatter="{:.1f}"), use_container_width=True, hide_index=True)
                            st.markdown("<hr style='margin: 0.8rem 0;'/>", unsafe_allow_html=True)

                    with col2:
                        for owner in owner_groups[1]:
                            owner_df = display_df[display_df["Team Owner"] == owner].drop(columns=["Team Owner"])
                            st.markdown(f"#### <span style='font-size: 1em;'>Team Owner: {owner}</span>", unsafe_allow_html=True)
                            st.dataframe(owner_df.style.set_properties(**{'font-size': '0.7em'}).format(subset=['Projection'], formatter="{:.1f}"), use_container_width=True, hide_index=True)
                            st.markdown("<hr style='margin: 0.8rem 0;'/>", unsafe_allow_html=True)
            else:
                st.info("No player data available for the selected match.")

        else:
            st.error("Match format error. Please check the selected match.")
    else:
        st.info("Select a match to analyze player impact.")
    st.markdown("*Concise player projections for the upcoming game!*")