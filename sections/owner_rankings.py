import streamlit as st
import pandas as pd
import numpy as np
import random

def show_rank(df: pd.DataFrame, df_diff: pd.DataFrame, points_df: pd.DataFrame, available_matches_df: pd.DataFrame, n_matches_played: int, total_matches: int, top4_count: dict):
    st.subheader("📊🏆 Owner Rankings: Current vs Predicted (Next Match Scores)", divider="orange")

    # --- Match Selection ---
    available_matches = available_matches_df["Match"].tolist()
    match_input = st.selectbox("Current/Next Match", options=available_matches, index=0)
    st.session_state.match_input = match_input

    if not match_input:
        st.error("No match selected.")
    else:
        teams_playing = [team.strip() for team in match_input.split("vs")]
        if len(teams_playing) != 2:
            st.error("Match format error. Please check the selected match.")
        else:
            team1, team2 = teams_playing
            exclusion_pattern = r"\(O\)|\(RE\)"
            match_players = sorted(
                points_df[
                    (points_df["Team"].isin([team1, team2])) &
                    (~points_df["Player Name"].str.contains(exclusion_pattern, regex=True))
                ]["Player Name"].unique()
            )

            non_playing_players = st.multiselect(
                f"Select Non-Playing Players from {team1}/{team2}:",
                options=match_players,
                key=f"non_playing_{match_input.replace(' ', '_')}" # Unique key per match
            )

            # --- Prediction Logic ---
            x = np.arange(len(df.columns[1:])).reshape(-1, 1)
            predictions = []

            for i, owner in enumerate(df["Owners"]):
                y = df.iloc[i, 1:].values.reshape(-1, 1)
                from sklearn.linear_model import LinearRegression
                model = LinearRegression().fit(x, y)
                last_score = y[-1][0]

                owner_players = points_df[
                    (points_df["Owner"] == owner) &
                    (points_df["Team"].isin(teams_playing)) &
                    (~points_df["Player Name"].str.contains(exclusion_pattern, regex=True)) &
                    (~points_df["Player Name"].isin(non_playing_players))
                ]

                avg_next_points = owner_players["Total Points"].sum() / n_matches_played if not owner_players.empty else 0
                predicted_next = last_score + avg_next_points
                change_pct = ((predicted_next - last_score) / last_score * 100) if last_score else 0
                top_appearance = top4_count[owner]

                predictions.append([
                    owner, last_score, int(predicted_next),
                    f"{change_pct:.1f}%", owner_players.shape[0], top_appearance
                ])

            merged_df = pd.DataFrame(predictions, columns=[
                "Owners", "Current Score", "Predicted Next Score",
                "Change (%)", "Players in Next Match", "Top 4 Appearances"
            ])

            # --- Winning Chances ---
            merged_df["Projected Final Score"] = merged_df["Current Score"] + \
                (merged_df["Predicted Next Score"] - merged_df["Current Score"]) * (total_matches - n_matches_played)
            total_projected = merged_df["Projected Final Score"].sum()
            merged_df["Winning Chances (%)"] = (merged_df["Projected Final Score"] / total_projected * 100).round(1)
            merged_df.drop(columns=["Projected Final Score"], inplace=True)

            # --- Ranking ---
            merged_df["Rank"] = merged_df["Current Score"].rank(method='first', ascending=False).astype(int)
            merged_df = merged_df.sort_values(by="Current Score", ascending=False).reset_index(drop=True)

            # --- Status Column (Q/E/blank) ---
            remaining_matches = total_matches - n_matches_played
            merged_df["Projected Points"] = (merged_df["Predicted Next Score"] - merged_df["Current Score"]) * remaining_matches
            merged_df["Max Score"] = merged_df["Current Score"] + merged_df["Projected Points"]
            cutoff_score = merged_df.sort_values("Current Score", ascending=False).iloc[3]["Current Score"]

            def determine_status(row):
                if row["Rank"] <= 4 and row["Current Score"] >= cutoff_score:
                    return "Q"
                elif row["Max Score"] >= cutoff_score:
                    return ""
                else:
                    return "E"

            merged_df["Status"] = merged_df.apply(determine_status, axis=1)
            merged_df.drop(columns=["Projected Points", "Max Score"], inplace=True)

            # --- Rank Deltas ---
            scores = merged_df["Current Score"].values
            def format_delta(val):
                return "" if val == "" else int(val) if val == int(val) else round(val, 1)
            next_deltas = [""] + [str(format_delta(scores[i-1] - scores[i])) for i in range(1, len(scores))]
            first_deltas = [str(format_delta(scores[0] - s)) if i != 0 else "" for i, s in enumerate(scores)]
            merged_df["Next Rank Delta"] = next_deltas
            merged_df["1st Rank Delta"] = first_deltas

            # --- Arrow Icons in Owners Column ---
            latest_col, prev_col = df.columns[-1], df.columns[-2]
            temp_df = df[["Owners", prev_col, latest_col]].copy()
            temp_df["Prev Rank"] = temp_df[prev_col].rank(method="first", ascending=False).astype(int)
            temp_df["Curr Rank"] = temp_df[latest_col].rank(method="first", ascending=False).astype(int)

            def rank_change_arrow(row):
                delta = row["Prev Rank"] - row["Curr Rank"]
                if delta > 0:
                    return f'🔥 (+{delta})'
                elif delta < 0:
                    return f'🔻 ({delta})'
                else:
                    return ''


            temp_df["Styled Arrow"] = temp_df.apply(rank_change_arrow, axis=1)
            arrow_map = dict(zip(temp_df["Owners"], temp_df["Styled Arrow"]))
            merged_df["Owners"] = merged_df["Owners"].apply(lambda x: f'{x} {arrow_map.get(x, "")}')

            # --- Points Gained Since Last Update ---
            latest_scores = dict(zip(df["Owners"], df[latest_col]))
            prev_scores = dict(zip(df["Owners"], df[prev_col]))
            points_gained = {owner: latest_scores[owner] - prev_scores[owner] for owner in df["Owners"]}
            merged_df["Points Gained"] = merged_df["Owners"].apply(lambda x: points_gained.get(x.split(" ")[0], 0))  # remove emoji if needed

            # --- Predict Final Tournament Score ---

            # Define remaining league match teams and qualified playoff teams
            remaining_league_teams = ["PBKS", "MI", "LSG", "RCB"]
            playoff_teams = ["RCB", "MI", "PBKS", "GT"]

            unavailable_playoff_players = [
                "Jos Buttler [C]", "Ryan Rickelton", "Will Jacks",
                "Marco Jansen", "Phil Salt [C]", "Tim David"
]

            def get_avg_points_for_teams(owner, teams, exclude_players=None):
                players = points_df[
                    (points_df["Owner"] == owner) &
                    (points_df["Team"].isin(teams)) &
                    (~points_df["Player Name"].str.contains(exclusion_pattern, regex=True)) &
                    (~points_df["Player Name"].isin(non_playing_players))
                ]
                if exclude_players:
                    players = players[~players["Player Name"].isin(exclude_players)]
                return players["Total Points"].sum() / n_matches_played if not players.empty else 0


            league_projected_points = []
            playoff_projected_points = []

            for owner in merged_df["Owners"]:
                owner_clean = owner.split(" 🔻")[0].split(" 🔥")[0].strip()

                league_avg = get_avg_points_for_teams(owner_clean, remaining_league_teams)
                playoff_avg = get_avg_points_for_teams(owner_clean, playoff_teams, exclude_players=unavailable_playoff_players)

                league_points = league_avg * 2
                playoff_points = playoff_avg * 2

                league_projected_points.append(round(league_points))
                playoff_projected_points.append(round(playoff_points))


            merged_df["Projected League Pts"] = league_projected_points
            merged_df["Projected Playoff Pts"] = playoff_projected_points
            merged_df["Predicted Final Tournament Score"] = (
                merged_df["Current Score"] + merged_df["Projected League Pts"] + merged_df["Projected Playoff Pts"]
            )
            + merged_df["Projected Playoff Pts"]



            # --- Final Column Reordering ---
            ordered_cols = [
                "Status", "Rank", "Owners", "Current Score",
                "Next Rank Delta", "1st Rank Delta", "Points Gained",
                "Predicted Next Score", "Change (%)",
                "Players in Next Match", "Top 4 Appearances", "Winning Chances (%)", "Predicted Final Tournament Score"
            ]
            merged_df = merged_df[ordered_cols]

            # --- Display Table ---
            st.dataframe(merged_df, use_container_width=True, hide_index=True)

            st.session_state.owner_ranking_df = merged_df.copy()

            # --- Add footer ---
            st.markdown("---")

    # --- Owner of the Match Highlight ---
    st.subheader("🏅 Owner of the Match",divider="orange")
    latest_col = df_diff.columns[-1]
    latest_diff = df_diff[["Owners", latest_col]].sort_values(by=latest_col, ascending=False)
    top_owner_row = latest_diff.iloc[0]
    st.success(f"🥇 {top_owner_row['Owners']} scored highest in the last match with {int(top_owner_row[latest_col])} points!")


    # --- Player Summary Messages ---
    def get_message(gained_points, owner):
        high_scorer_msgs = [
            "🔥 {owner} had an excellent match scoring {points} points! Keep it up!",
            "🏆 {owner} absolutely crushed it with {points} points!",
            "🌟 {owner} delivered a top-tier performance with {points} points. Bravo!",
            "💪 {owner} led the scoreboard this match with {points} points. Keep shining!",
        ]
        average_scorer_msgs = [
            "✅ {owner} performed decently with {points} points.",
            "👏 {owner} made a solid contribution of {points} points.",
            "📈 {owner} added a fair {points} points to the tally.",
        ]
        low_scorer_msgs = [
            "📉 {owner} got only {points} points this time. A comeback is needed!",
            "😬 {owner} struggled a bit, managing just {points} points.",
            "🫤 {owner} couldn’t do much this time. Only {points} points on the board.",
        ]
        zero_scorer_msgs = [
            "😐 {owner} couldn't score this match. Hoping for a better show next time!",
            "🔇 {owner} was silent on the scoreboard this time. Next match is yours!",
            "🛑 {owner} got no points. Let’s bounce back stronger!",
        ]

        if gained_points >= 100:
            msg = random.choice(high_scorer_msgs)
        elif 50 <= gained_points < 100:
            msg = random.choice(average_scorer_msgs)
        elif 1 <= gained_points < 50:
            msg = random.choice(low_scorer_msgs)
        else:
            msg = random.choice(zero_scorer_msgs)

        return msg.format(owner=owner, points=gained_points)


    # Generate messages dynamically from df_diff
    ticker_messages = []
    latest_col_diff = df_diff.columns[-1]
    for index, row in df_diff.iterrows():
        owner = row["Owners"]
        gained_points = int(row[latest_col_diff])
        ticker_messages.append(get_message(gained_points, owner))

    ticker_text = " <span style='color: #ff9800;'>|</span> ".join(ticker_messages)


    ticker_html = f"""
    <div id="ticker-container" style="
        width: 100%;
        overflow: hidden;
        background-color: #111;
        padding: 10px 0;
        border-radius: 8px;
        border: 1px solid #444;
    ">
        <div id="scrolling-text" style="
            display: inline-block;
            white-space: nowrap;
            animation: scroll-left 40s linear infinite;
        ">
            <span style="font-size: 18px; font-weight: 500; color: #f8f8f2;">
                {ticker_text}    {ticker_text}
            </span>
        </div>
    </div>

    <style>
    @keyframes scroll-left {{
        0% {{
            transform: translateX(0%);
        }}
        100% {{
            transform: translateX(-50%);
        }}
    }}
    </style>
    """
    st.markdown(ticker_html, unsafe_allow_html=True)