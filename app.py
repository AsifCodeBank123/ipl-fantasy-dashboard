import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import pytz
import random
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
# --- Additional Imports for Live Scores ---
import lxml
import requests
from bs4 import BeautifulSoup
from sections.owner_insights import show_owner_insights
from sections.owners_performance import show_owners_performance
#from sections.trades import mini_auction
#from sections.c_vc_optimize show_cvc



if "match_input" not in st.session_state:
    st.session_state.match_input = None

if "non_playing_players" not in st.session_state:
    st.session_state.non_playing_players = []

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="HPL Fantasy Dashboard")
st.header("🏏 :orange[HPL] Fantasy League Performance Dashboard", divider = "orange")

# --- Inputs ---
n_matches_played = 10
total_matches = 14

# --- Captain and Vice-Captain selections for each owner ---
captain_vc_dict = {
    "Mahesh": ("Jos Buttler[C]", "Sai Sudarsan[VC]"),
    "Asif": ("Kuldeep Yadav[C]", "Pat Cummins[VC]"),
    "Pritesh": ("Abhishek Sharma[C]", "Yashasvi Jaiswal[VC]"),
    "Pritam": ("Suryakumar Yadav[C]", "Virat Kohli[VC]"),
    "Lalit": ("Shreyas Iyer[C]", "Shubman Gill[VC]"),
    "Umesh": ("Travis Head[C]", "Rajat Patidar[VC]"),
    "Sanskar": ("Hardik Pandya[C]", "Nicholas Pooran[VC]"),
    "Johnson": ("Sunil Naraine[C]", "Mitchell Starc[VC]"),
    "Somansh": ("Phil Salt[C]","Rashid Khan[VC]"),
    "Wilfred": ("KL Rahul[C]","Rachin Ravindra[VC]", )
}


match_data = pd.read_csv("match_schedule.csv")
# Convert match data into a DataFrame for easier manipulation
match_df = pd.DataFrame(match_data, columns=["Date", "Time", "Match"])

# Extract team codes from Match column
scheduled_teams = set()
for match in match_df["Match"]:
    teams = match.split(" vs ")
    scheduled_teams.update(teams)


# --- Maintain a finished matches set globally ---
finished_matches = set()

def fetch_live_matches():
    global finished_matches

    try:
        link = "https://www.cricbuzz.com/cricket-match/live-scores"
        source = requests.get(link, timeout=5).text
        page = BeautifulSoup(source, "lxml")

        main_section = page.find("div", class_="cb-col cb-col-100 cb-bg-white")
        matches = main_section.find_all("div", class_="cb-scr-wll-chvrn cb-lv-scrs-col")

        live_matches = []
        current_time = datetime.now(ist)

        for match in matches:
            match_text = match.text.strip()

            # Attempt to split score and status
            if " - " in match_text:
                score_part, status_part = match_text.split(" - ", 1)
            else:
                score_part, status_part = match_text, ""

            # Check if match is finished
            if "won" in status_part.lower():
                # Mark this match as finished
                finished_matches.add(score_part.strip())
                continue  # Skip fetching finished match

            # Check against available matches
            for scheduled_match in available_matches_df["Match"]:
                teams = scheduled_match.split(" vs ")
                if any(team in match_text for team in teams):
                    if scheduled_match not in finished_matches:
                        live_matches.append(match_text)
                    break  # Once matched, no need to check further

        return live_matches

    except Exception as e:
        return [f"⚠️ Failed to load live matches: {str(e)}"]

# --- Load Data ---
df = pd.read_csv("owners_performance_updates.csv")
points_df = pd.read_csv("points.csv")

# Optional: wrap column headers or shorten names in your DataFrame
points_df.columns = [col if len(col) < 15 else col[:12] + "..." for col in points_df.columns]

ist = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(ist)

# Combine Date and Time into a single DateTime column
match_df['DateTime'] = match_df['Date'] + ' ' + match_df['Time']
match_df['DateTime'] = pd.to_datetime(match_df['DateTime'], format='%d-%b-%y %I:%M %p').dt.tz_localize("Asia/Kolkata")

# --- Helper function to get available matches ---
def get_available_matches(match_df):
    current_time = datetime.now(ist)  # Move this inside
    match_df = match_df.copy()
    match_df["MatchStartWindow"] = match_df["DateTime"] - timedelta(minutes=30)
    match_df["MatchEndWindow"] = match_df["DateTime"] + timedelta(hours=4)

    available_matches_df = match_df[
        ((current_time >= match_df["MatchStartWindow"]) & (current_time <= match_df["MatchEndWindow"])) |
        (match_df["DateTime"] > current_time)
    ].copy()

    available_matches_df = available_matches_df.sort_values("DateTime")
    return available_matches_df


# --- Setup: Get available matches once ---
#current_time = datetime.now(ist)
available_matches_df = get_available_matches(match_df)

# You can now use available_matches_df anywhere below without recalculating

# Ensure the "CVC Bonus Points" column exists and is of float type
if "CVC Bonus Points" not in points_df.columns:
    points_df["CVC Bonus Points"] = 0.0
else:
    points_df["CVC Bonus Points"] = points_df["CVC Bonus Points"].astype(float)

# Apply captain and vice-captain bonus points
for owner, (captain, vice_captain) in captain_vc_dict.items():
    for role, multiplier in zip([captain, vice_captain], [1.0, 1.0]):  # set back to 2.0, 1.5 when ready
        mask = (points_df["Owner"] == owner) & (points_df["Player Name"] == role)
        points_df.loc[mask, "CVC Bonus Points"] = points_df.loc[mask, "Total Points"] * multiplier


# --- Calculate Update Differences ---
df_diff = df.copy()
df_diff.iloc[:, 2:] = df.iloc[:, 2:].diff(axis=1).fillna(df.iloc[:, 2:3])
latest_col = df_diff.columns[-1]

# --- Top 4 Appearance Count ---
top4_count = {owner: 0 for owner in df["Owners"]}
for idx, update in enumerate(df.columns[1:]):
    if idx == 0:
        top_gainers = df.nlargest(4, update)
    else:
        top_gainers = df_diff.nlargest(4, update)
    for owner in top_gainers["Owners"]:
        top4_count[owner] += 1

st.session_state.top4_count = top4_count
st.session_state.captain_vc_dict = captain_vc_dict

# Sidebar navigation
with st.sidebar.expander("📂 Select Section", expanded=True):
    section=st.radio("",[
        "Owner Rankings: Current vs Predicted",
        "Player Impact - Next Match Focus",
        "Team vs Team Comparison",
        "Team of the Tournament",
        "Owner Insights & Breakdown",
        "Owners Performance"
    ])


def format_live_match(match_text):
    import re

    result = ""

    # Remove leading ℹ️ if present
    match_text = match_text.replace("ℹ️", "").strip()

    # 1. Find all "TEAM + SCORE" like MI12-0 (1.3 Ovs)
    team_score_pattern = r'([A-Z]{2,4})(\d+-\d+ \(\d+(\.\d+)? Ovs\))'
    matches = re.findall(team_score_pattern, match_text)

    for match in matches:
        team_code, score, _ = match
        result += f"🏏 **{team_code}** - {score}\n"

    # Remove matched part from original text
    for match in matches:
        full_match = "".join(match[:-1])  # exclude last group (decimal part)
        match_text = match_text.replace(full_match, "").strip()

    # 2. Now look for any TEAM codes separately (like LSG)
    words = match_text.split()

    for word in words:
        if word in scheduled_teams:
            result += f"🏏 **{word}**\n"
            match_text = match_text.replace(word, "").strip()

    # 3. Remaining text is status update
    if match_text.strip():
        result += f"\nℹ️ {match_text.strip()}\n"

    return result



# --- Streamlit Display ---
st.markdown("### 📺 Live Score")
st.write("")

live_scores = fetch_live_matches()

if live_scores:
    for match in live_scores:
        formatted_text = format_live_match(match)
        st.markdown(formatted_text)  # <-- USE MARKDOWN for line breaks!
        st.write("")  # Space between matches
else:
    st.info("No live matches at the moment.")

st.write("")


if section == "Owner Rankings: Current vs Predicted":
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
                options=match_players
            )

            # --- Prediction Logic ---
            x = np.arange(len(df.columns[1:])).reshape(-1, 1)
            predictions = []

            for i, owner in enumerate(df["Owners"]):
                y = df.iloc[i, 1:].values.reshape(-1, 1)
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
                if row["Rank"] <= 2 and row["Current Score"] >= cutoff_score:
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
                if row["Curr Rank"] < row["Prev Rank"]:
                    return '🚀'
                elif row["Curr Rank"] > row["Prev Rank"]:
                    return '🔻'
                else:
                    return '➡'

            temp_df["Styled Arrow"] = temp_df.apply(rank_change_arrow, axis=1)
            arrow_map = dict(zip(temp_df["Owners"], temp_df["Styled Arrow"]))
            merged_df["Owners"] = merged_df["Owners"].apply(lambda x: f'{x} {arrow_map.get(x, "")}')

            # --- Final Column Reordering ---
            ordered_cols = [
                "Status", "Rank", "Owners", "Current Score",
                "Next Rank Delta", "1st Rank Delta",
                "Predicted Next Score", "Change (%)",
                "Players in Next Match", "Top 4 Appearances", "Winning Chances (%)"
            ]
            merged_df = merged_df[ordered_cols]

            # --- Display Table ---
            st.dataframe(merged_df, use_container_width=True, hide_index=True)

            # --- Add footer ---
            st.markdown("---")

    # --- Owner of the Match Highlight ---
    st.subheader("🏅 Owner of the Match",divider="orange")
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
    for index, row in df_diff.iterrows():
        owner = row["Owners"]
        gained_points = int(row[latest_col])
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


elif section == "Player Impact - Next Match Focus":
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
                options=non_playing_options
            )

            # Set playing status (still needed for filtering, even if not displayed)
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

            # Final display columns with renamed columns and removed "Total Points" and "Playing Status"
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
        st.info("Select a match to analyze player impact.").markdown("*Concise player projections for the upcoming game!*")


elif section == "Team vs Team Comparison":
    st.subheader("🧠 Team vs Team Comparison - Owner's Team", divider="orange")

    # --- Select Owners ---
    col1, col2 = st.columns(2)
    with col1:
        owner1 = st.selectbox("Select Owner 1", sorted(points_df["Owner"].unique()), key="owner1")
    with col2:
        owner2 = st.selectbox("Select Owner 2", sorted(points_df["Owner"].unique()), key="owner2")

    # --- Filter Data ---
    owner1_df = points_df[points_df["Owner"] == owner1]
    owner2_df = points_df[points_df["Owner"] == owner2]

    # --- Basic Stats ---
    def get_summary(df):
        return {
            "Total Players": df.shape[0],
            "Total Points": int(df["Total Points"].sum()),
            "Average Points": round(df["Total Points"].mean(), 1),
            "Total Value": int(df["Player Value"].sum())
        }

    owner1_stats = get_summary(owner1_df)
    owner2_stats = get_summary(owner2_df)

    # --- Display Summary ---
    st.markdown("### 🧾 Team Summary")
    summary_df = pd.DataFrame([owner1_stats, owner2_stats], index=[owner1, owner2])
    st.dataframe(summary_df, use_container_width=True)

    # --- Top Players Side by Side ---
    st.markdown("### 🥇 Top 5 Players Comparison")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"**{owner1}'s Top Players**")
        st.dataframe(owner1_df.sort_values("Total Points", ascending=False).head(5)[["Player Name", "Team", "Total Points"]], use_container_width=True)
    with col4:
        st.markdown(f"**{owner2}'s Top Players**")
        st.dataframe(owner2_df.sort_values("Total Points", ascending=False).head(5)[["Player Name", "Team", "Total Points"]], use_container_width=True)

    # --- Pie Chart: Team Composition ---
    st.markdown("### 🧩 Team Composition Breakdown (By IPL Team)")
    team_comp1 = owner1_df.groupby("Team")["Total Points"].sum().reset_index()
    team_comp2 = owner2_df.groupby("Team")["Total Points"].sum().reset_index()

    col5, col6 = st.columns(2)
    with col5:
        fig1 = px.pie(team_comp1, names="Team", values="Total Points", title=f"{owner1} - Team Split", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig1, use_container_width=True, key=f"owner1_team_split")
    with col6:
        fig2 = px.pie(team_comp2, names="Team", values="Total Points", title=f"{owner2} - Team Split", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig2, use_container_width=True, key=f"owner2_team_split")

    # --- Common Players ---
    common_players = pd.merge(
        owner1_df[["Player Name"]],
        owner2_df[["Player Name"]],
        on="Player Name"
    )

    if not common_players.empty:
        st.markdown("### 🔁 Shared Players")
        st.dataframe(common_players, use_container_width=True, hide_index=True)
    else:
        st.info("No shared players between the selected owners.")


    # --- Who is Better? ---
    st.markdown("### 🧠 Verdict: Who Has the Stronger Team?")

    # Simple weighted score: 50% total points, 30% avg points, 20% top player's score
    owner1_top_score = owner1_df["Total Points"].max()
    owner2_top_score = owner2_df["Total Points"].max()

    score1 = (owner1_stats["Total Points"] * 0.5 +
              owner1_stats["Average Points"] * 0.3 +
              owner1_top_score * 0.2)
    score2 = (owner2_stats["Total Points"] * 0.5 +
              owner2_stats["Average Points"] * 0.3 +
              owner2_top_score * 0.2)

    if score1 > score2:
        st.success(f"✅ Based on performance metrics, **{owner1}** currently has the stronger team!")
    elif score2 > score1:
        st.success(f"✅ Based on performance metrics, **{owner2}** currently has the stronger team!")
    else:
        st.info("🤝 It's a tie! Both teams are evenly matched.")

    st.markdown("### ⭐ Owner Performance Ratings Comparison")

    # --- Rating Section ---
    rating_rows_comparison = []

    
    top4_count = st.session_state.get("top4_count")
    captain_vc_dict = st.session_state.get("captain_vc_dict")

    if top4_count is not None and captain_vc_dict is not None:
        def calculate_owner_rating(owner):
            total_pts = df.loc[df["Owners"] == owner].iloc[:, -1].values[0]
            gain = df.loc[df["Owners"] == owner].iloc[:, -1] - df.loc[df["Owners"] == owner].iloc[:, -2]
            captain = captain_vc_dict.get(owner, ("", ""))[0]
            vc = captain_vc_dict.get(owner, ("", ""))[1]
            captain_pts = points_df.loc[(points_df["Owner"] == owner) & (points_df["Player Name"] == captain), "Total Points"].sum()
            vc_pts = points_df.loc[(points_df["Owner"] == owner) & (points_df["Player Name"] == vc), "Total Points"].sum()
            cvc_total_bonus = points_df.loc[points_df["Owner"] == owner, "CVC Bonus Points"].sum()
            team_total = points_df.loc[points_df["Owner"] == owner, "Total Points"].sum()
            #win_chance = merged_df.loc[merged_df["Owners"] == owner, "Winning Chances (%)"].values[0]
            top4 = top4_count.get(owner, 0)
            max_top4 = max(top4_count.values()) if top4_count else 1

            score = (
                (total_pts / df.iloc[:, -1].max()) * 10 * 0.2 +
                (gain.values[0] / 50) * 10 * 0.1 +
                (captain_pts / points_df["Total Points"].max() if not points_df.empty else 0) * 10 * 0.15 +
                (vc_pts / points_df["Total Points"].max() if not points_df.empty else 0) * 10 * 0.1 +
                (cvc_total_bonus / team_total if team_total > 0 else 0) * 10 * 0.15 +
                #(win_chance / 100) * 10 * 0.2 +
                (top4 / max_top4 if max_top4 > 0 else 0) * 10 * 0.1
            )
            num_stars = round(score / 2)
            stars = "⭐" * int(max(0, min(5, num_stars)))
            return round(score, 1), stars

        score_owner1, stars_owner1 = calculate_owner_rating(owner1)
        rating_rows_comparison.append([owner1, score_owner1, stars_owner1])

        score_owner2, stars_owner2 = calculate_owner_rating(owner2)
        rating_rows_comparison.append([owner2, score_owner2, stars_owner2])

        rating_df_comparison = pd.DataFrame(rating_rows_comparison, columns=["Owner", "Performance Score (out of 10)", "⭐ Rating"])
        st.dataframe(rating_df_comparison, use_container_width=True)
    else:
        st.warning("Owner rating data is not available for comparison.")


elif section == "Team of the Tournament":

    st.markdown("## 🏆 Team of the Tournament")

    # Exclude Out/Released players
    valid_players_df = points_df[
        ~points_df["Player Name"].str.contains(r"\(O\)|\(R\)|\(RE\)", na=False)
    ]

    # Get top 11 scorers
    top_players = valid_players_df.sort_values(by="Total Points", ascending=False).drop_duplicates(subset=["Player Name"]).head(11).reset_index(drop=True)

    # Assign captain and vice-captain
    captain = top_players.loc[0, "Player Name"]
    vice_captain = top_players.loc[1, "Player Name"]

    # Layout for visual structure
    cols = st.columns(3)
    positions = ['Forward', 'Midfielder', 'Defender']  # Visual layout only

    for i, player in top_players.iterrows():
        col = cols[i % 3]
        with col:
            st.markdown(f"### {player['Player Name']}")
            st.markdown(f"**Team:** {player['Team']}")
            st.markdown(f"**Points:** {player['Total Points']}")
            
            # Highlight captain and VC
            if player["Player Name"] == captain:
                st.markdown("🧢 **Captain**")
            elif player["Player Name"] == vice_captain:
                st.markdown("⭐️ **Vice-Captain**")

            st.markdown("---")

# elif section == "Captain/Vice-Captain Impact Analysis":
#     st.subheader("💥 Captain/Vice-Captain Impact Analysis", divider="orange")

#     summary = points_df.groupby("Owner").agg(
#         Team_Total_Points=("Total Points", "sum"),
#         CVC_Bonus_Points=("CVC Bonus Points", "sum")
#     ).reset_index()

#     summary["CVC_Impact_%"] = (summary["CVC_Bonus_Points"] / summary["Team_Total_Points"]) * 100
#     summary = summary.sort_values("CVC_Bonus_Points", ascending=False)

#     st.dataframe(
#         summary.style.format({
#             "CVC_Impact_%": "{:.0f}%", 
#             "CVC_Bonus_Points": "{:.0f}", 
#             "Team_Total_Points": "{:.0f}"
#         }),
#         use_container_width=True
#     )

# elif section == "Best C/VC Suggestion":
      #show_cvc(points_df)


# elif section == "Players to Watch Out for in Mini Auction":
#     mini_auction(points_df)

elif section == "Owner Insights & Breakdown":
    show_owner_insights(points_df)

elif section == "Owners Performance":
   show_owners_performance(df)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
