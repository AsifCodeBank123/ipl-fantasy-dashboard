import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import pytz
import re

# --- Additional Imports for Live Scores ---
import requests
from bs4 import BeautifulSoup

from sections.owner_insights import show_owner_insights
from sections.owners_performance import show_owners_performance
#from sections.trades import mini_auction
#from sections.c_vc_optimize show_cvc
from sections.team_comparison import show_comparison
from sections.team_of_tournament import show_team
from sections.player_impact import show_impact
from sections.owner_rankings import show_rank
from sections.qualification_chances import show_chances


if "match_input" not in st.session_state:
    st.session_state.match_input = None

if "non_playing_players" not in st.session_state:
    st.session_state.non_playing_players = []

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="HPL Fantasy Dashboard")
st.header("🏏 :orange[HPL] Fantasy League Performance Dashboard", divider = "orange")

# --- Inputs ---
n_matches_played = 11
total_matches = 14

# --- Captain and Vice-Captain selections for each owner ---
captain_vc_dict = {
    "Mahesh": ("Jos Buttler [C]", "B. Sai Sudharsan [VC]"),
    "Asif": ("Kuldeep Yadav [C]", "Pat Cummins [VC]"),
    "Pritesh": ("Abhishek Sharma [C]", "Yashasvi Jaiswal [VC]"),
    "Pritam": ("Suryakumar Yadav [C]", "Virat Kohli [VC]"),
    "Lalit": ("Shreyas Iyer [C]", "Shubman Gill [VC]"),
    "Umesh": ("Travis Head [C]", "Rajat Patidar [VC]"),
    "Sanskar": ("Hardik Pandya [C]", "Nicholas Pooran [VC]"),
    "Johnson": ("Sunil Naraine [C]", "Mitchell Starc [VC]"),
    "Somansh": ("Phil Salt [C]","Rashid Khan [VC]"),
    "Wilfred": ("KL Rahul [C]","Rachin Ravindra [VC]", )
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

# --- Helper function to get available matches ---
def get_available_matches(match_df):
    ist = pytz.timezone("Asia/Kolkata")
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

# Combine Date and Time into a single DateTime column
match_df['DateTime'] = match_df['Date'] + ' ' + match_df['Time']
match_df['DateTime'] = pd.to_datetime(match_df['DateTime'], format='%d-%b-%y %I:%M %p').dt.tz_localize("Asia/Kolkata")

# --- Setup: Get available matches once ---
available_matches_df = get_available_matches(match_df) # <--- Call it here

def fetch_live_matches():
    global finished_matches

    try:
        link = "https://www.cricbuzz.com/cricket-match/live-scores"
        source = requests.get(link, timeout=5).text
        page = BeautifulSoup(source, "lxml")

        main_section = page.find("div", class_="cb-col cb-col-100 cb-bg-white")
        matches = main_section.find_all("div", class_="cb-scr-wll-chvrn cb-lv-scrs-col")

        live_matches = []

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
                    break   # Once matched, no need to check further

        return live_matches

    except Exception as e:
        return [f"⚠️ Failed to load live matches: {str(e)}"]


def format_live_match(match_text):

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

    # Now look for any TEAM codes separately (like LSG)
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

# --- Load Data ---
df = pd.read_csv("owners_performance_updates.csv")
points_df = pd.read_csv("points.csv")

# Optional: wrap column headers or shorten names in your DataFrame
points_df.columns = [col if len(col) < 15 else col[:12] + "..." for col in points_df.columns]

ist = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(ist)


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
        "Owners Performance",
        "Qualification Chances"
    ])

if section == "Owner Rankings: Current vs Predicted":
    show_rank(df, df_diff, points_df, available_matches_df, n_matches_played, total_matches, st.session_state.top4_count)


elif section == "Player Impact - Next Match Focus":
    show_impact(points_df, available_matches_df, n_matches_played)

elif section == "Team vs Team Comparison":
    show_comparison(points_df, df)


elif section == "Team of the Tournament":
    show_team(points_df)

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
elif section == "Qualification Chances":
    show_chances(points_df, match_data, n_matches_played)


elif section == "Owner Insights & Breakdown":
    show_owner_insights(points_df)

elif section == "Owners Performance":
   show_owners_performance(df)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
