import streamlit as st
import pandas as pd
import numpy as np


def show_comparison(points_df, df, captain_vc_dict, top4_count):
    st.header("Team vs Team Comparison")

    teams = sorted(df["Team"].dropna().unique())
    team1 = st.selectbox("Select Team 1", teams, key="team1")
    team2 = st.selectbox("Select Team 2", teams, key="team2")

    if team1 == team2:
        st.warning("Please select two different teams.")
        return

    team1_players = df[df["Team"] == team1]["Player Name"].tolist()
    team2_players = df[df["Team"] == team2]["Player Name"].tolist()

    def get_owner_points(players, points_df):
        owner_scores = {}
        for _, row in points_df.iterrows():
            owner = row["Owner"]
            score = 0
            for player in players:
                if player in row["Player Name"]:
                    score += row["Total Points"]
            owner_scores[owner] = score
        return owner_scores

    team1_owner_points = get_owner_points(team1_players, points_df)
    team2_owner_points = get_owner_points(team2_players, points_df)

    owner_comparison = []
    for owner in points_df["Owner"].unique():
        team1_pts = team1_owner_points.get(owner, 0)
        team2_pts = team2_owner_points.get(owner, 0)
        diff = team1_pts - team2_pts
        result = "Team 1 better" if diff > 0 else "Team 2 better" if diff < 0 else "Equal"
        owner_comparison.append({
            "Owner": owner,
            f"{team1} Points": team1_pts,
            f"{team2} Points": team2_pts,
            "Difference": diff,
            "Better Team": result
        })

    df_comparison = pd.DataFrame(owner_comparison)
    st.dataframe(df_comparison.sort_values("Difference", ascending=False), use_container_width=True)

    team1_total = df[df["Team"] == team1]["Total Points"].sum()
    team2_total = df[df["Team"] == team2]["Total Points"].sum()
    st.markdown(f"### Total Points: {team1}: `{team1_total}` | {team2}: `{team2_total}`")

    # Optional: Top contributors
    st.subheader("Top Contributors")
    top_contributors = df[df["Team"].isin([team1, team2])].sort_values("Total Points", ascending=False).head(top4_count)
    st.dataframe(top_contributors[["Player Name", "Team", "Total Points"]], use_container_width=True)
