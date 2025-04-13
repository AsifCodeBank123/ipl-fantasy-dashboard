import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import pytz
import random
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="HPL Fantasy Dashboard")
st.title("🏏 HPL Fantasy League Performance Dashboard")

# --- Captain and Vice-Captain selections for each owner ---
captain_vc_dict = {
    "Mahesh": ("Jos Buttler", "N. Tilak Varma"),
    "Asif": ("Pat Cummins", "Venkatesh Iyer"),
    "Pritesh": ("Abhishek Sharma", "Yashasvi Jaiswal"),
    "Pritam": ("Suryakumar Yadav", "Virat Kohli"),
    "Lalit": ("Shreyas Iyer", "Shubman Gill"),
    "Umesh": ("Travis Head", "Rohit Sharma"),
    "Sanskar": ("Hardik Pandya", "Axar Patel"),
    "Johnson": ("Sunil Naraine", "Sanju Samson"),
    "Somansh": ("Rashid Khan", "Phil Salt"),
    "Wilfred": ("Rachin Ravindra", "KL Rahul")
}


# --- Load Match Data ---
# Match data (schedule) for the IPL season.
match_data = [
    ["12-Apr-25", "3:30 PM", "LSG vs GT"],
    ["12-Apr-25", "7:30 PM", "SRH vs PK"],
    ["13-Apr-25", "3:30 PM", "RR vs RCB"],
    ["13-Apr-25", "7:30 PM", "DC vs MI"],
    ["14-Apr-25", "7:30 PM", "LSG vs CSK"],
    ["15-Apr-25", "7:30 PM", "KKR vs PK"],
    ["16-Apr-25", "7:30 PM", "DC vs RR"],
    ["17-Apr-25", "7:30 PM", "MI vs SRH"],
    ["18-Apr-25", "7:30 PM", "RCB vs PK"],
    ["19-Apr-25", "3:30 PM", "GT vs DC"],
    ["19-Apr-25", "7:30 PM", "LSG vs RR"],
    ["20-Apr-25", "3:30 PM", "PK vs RCB"],
    ["20-Apr-25", "7:30 PM", "MI vs CSK"],
    ["21-Apr-25", "7:30 PM", "KKR vs GT"],
    ["22-Apr-25", "7:30 PM", "LSG vs DC"],
    ["23-Apr-25", "7:30 PM", "SRH vs MI"],
    ["24-Apr-25", "7:30 PM", "RCB vs RR"],
    ["25-Apr-25", "7:30 PM", "CSK vs SRH"],
    ["26-Apr-25", "7:30 PM", "KKR vs PK"],
    ["27-Apr-25", "7:30 PM", "DC vs LSG"],
    ["28-Apr-25", "7:30 PM", "RR vs GT"],
    ["29-Apr-25", "7:30 PM", "DC vs RCB"],
    ["30-Apr-25", "7:30 PM", "DC vs KKR"],
    ["1-May-25", "7:30 PM", "CSK vs PK"],
    ["2-May-25", "7:30 PM", "RR vs MI"],
    ["3-May-25", "7:30 PM", "GT vs SRH"],
    ["4-May-25", "7:30 PM", "RCB vs CSK"],
    ["5-May-25", "7:30 PM", "KKR vs RR"],
    ["6-May-25", "7:30 PM", "PK vs LSG"],
    ["7-May-25", "7:30 PM", "SRH vs DC"],
    ["8-May-25", "7:30 PM", "MI vs GT"],
    ["9-May-25", "7:30 PM", "KKR vs CSK"],
    ["10-May-25", "7:30 PM", "PK vs DC"],
    ["11-May-25", "7:30 PM", "LSG vs RCB"],
    ["12-May-25", "7:30 PM", "SRH vs KKR"],
    ["13-May-25", "7:30 PM", "PK vs MI"],
    ["14-May-25", "7:30 PM", "DC vs GT"],
    ["15-May-25", "7:30 PM", "CSK vs RR"],
    ["16-May-25", "7:30 PM", "RCB vs SRH"],
    ["17-May-25", "7:30 PM", "GT vs LSG"],
    ["18-May-25", "3:30 PM", "MI vs DC"],
    ["18-May-25", "7:30 PM", "RR vs PK"],
    ["19-May-25", "7:30 PM", "RCB vs KKR"],
    ["20-May-25", "7:30 PM", "GT vs CSK"],
    ["21-May-25", "7:30 PM", "LSG vs SRH"],
    # more matches can be added
]

# Convert match data into a DataFrame for easier manipulation
match_df = pd.DataFrame(match_data, columns=["Date", "Time", "Match"])

# --- Load Data ---
df = pd.read_csv("owners_performance_updates.csv")
points_df = pd.read_csv("points.csv")

# Ensure the "CVC Bonus Points" column exists and is of float type
if "CVC Bonus Points" not in points_df.columns:
    points_df["CVC Bonus Points"] = 0.0
else:
    points_df["CVC Bonus Points"] = points_df["CVC Bonus Points"].astype(float)

# Apply captain and vice-captain multipliers
for owner, (captain, vice_captain) in captain_vc_dict.items():
    # Apply captain bonus
    points_df.loc[
        (points_df["Owner"] == owner) & (points_df["Player Name"] == captain),
        "CVC Bonus Points"
    ] = (
        points_df.loc[
            (points_df["Owner"] == owner) & (points_df["Player Name"] == captain),
            "Total Points"
        ] #* 2.0
    )

    # Apply vice-captain bonus
    points_df.loc[
        (points_df["Owner"] == owner) & (points_df["Player Name"] == vice_captain),
        "CVC Bonus Points"
    ] = (
        points_df.loc[
            (points_df["Owner"] == owner) & (points_df["Player Name"] == vice_captain),
            "Total Points"
        ] #* 1.5
    )



# Define IST timezone
ist = pytz.timezone("Asia/Kolkata")
# --- Get current time ---
current_time = datetime.now(ist)

# --- Convert match dates and times to datetime format in IST ---
match_df["DateTime"] = pd.to_datetime(
    match_df["Date"] + " " + match_df["Time"], format="%d-%b-%y %I:%M %p"
)
match_df["DateTime"] = match_df["DateTime"].dt.tz_localize("Asia/Kolkata")

# --- Filter matches after the current time ---
upcoming_matches_df = match_df[match_df["DateTime"] > current_time].copy()

# --- Calculate time difference ---
upcoming_matches_df["TimeDiff"] = upcoming_matches_df["DateTime"] - current_time
upcoming_matches_df["TimeDiffInHours"] = upcoming_matches_df["TimeDiff"].dt.total_seconds() / 3600

# --- Filter for matches within the next 4 hours ---
matches_within_4_hours = upcoming_matches_df[upcoming_matches_df["TimeDiffInHours"] <= 4]

# --- Select next match ---
if not matches_within_4_hours.empty:
    next_match = matches_within_4_hours.iloc[0]["Match"]
elif not upcoming_matches_df.empty:
    next_match = upcoming_matches_df.iloc[0]["Match"]
else:
    next_match = "No upcoming match"


# --- Inputs ---
n_matches_played = 5
total_matches = 10

# --- Dropdown of upcoming matches ---
upcoming_matches = upcoming_matches_df["Match"].tolist()

# Add next match to the dropdown
match_input = st.selectbox("Current/Next Match", options=upcoming_matches)

# Ensure match_input is valid before using it
if match_input:
    # Split the selected match into two teams
    teams_playing = match_input.split(" vs ")

    # Validate that teams_playing has exactly two elements
    if len(teams_playing) == 2:
        # Filter players based on the selected teams
        playing_players = points_df[points_df["Team"].isin(teams_playing)]["Player Name"].unique().tolist()
        playing_players.sort()

        # Dynamically update the multiselect for non-playing players from selected teams
        non_playing_players = st.multiselect(
            f"Select Non-Playing Players from {teams_playing[0]}/{teams_playing[1]}:",
            options=playing_players,
            default=[]
        )
    else:
        st.error("Error: Match format is incorrect or incomplete.")
else:
    st.error("No match selected. Please select a match.")


# --- Calculate Update Differences ---
df_diff = df.copy()
df_diff.iloc[:, 2:] = df.iloc[:, 2:].diff(axis=1).fillna(df.iloc[:, 2:3])
latest_col = df_diff.columns[-1]

# --- Owner of the Match Highlight ---
st.subheader("🏅 Owner of the Match")
latest_diff = df_diff[["Owners", latest_col]].sort_values(by=latest_col, ascending=False)
top_owner_row = latest_diff.iloc[0]
st.success(f"🥇 {top_owner_row['Owners']} scored highest in the last match with {int(top_owner_row[latest_col])} points!")

# --- Player Impact Table ---
st.subheader("🧠 Player Impact - Next Match Focus")
impact_df = points_df[(points_df["Team"].isin(teams_playing)) & (~points_df["Player Name"].isin(non_playing_players))]
top_players = impact_df.sort_values(by="Total Points", ascending=False).head(10)
st.dataframe(top_players[["Player Name", "Team", "Owner", "Total Points"]], use_container_width=True)

# Group by owner
owner_cvc_summary = points_df.groupby("Owner").agg(
    Team_Total_Points=("Total Points", "sum"),
    CVC_Bonus_Points=("CVC Bonus Points", "sum")
).reset_index()

owner_cvc_summary["CVC_Impact_%"] = (owner_cvc_summary["CVC_Bonus_Points"] / owner_cvc_summary["Team_Total_Points"]) * 100
owner_cvc_summary = owner_cvc_summary.sort_values(by="CVC_Bonus_Points", ascending=False)

st.subheader("💥 Captain/Vice-Captain Impact Analysis")
st.dataframe(owner_cvc_summary.style.format({"CVC_Impact_%": "{:.0f}%", "CVC_Bonus_Points": "{:.0f}", "Team_Total_Points": "{:.0f}"}))

st.subheader("🔮 What-If Best C/VC Optimization")

what_if_results = []

for owner, group in points_df.groupby("Owner"):
    owner_players = group.copy()

    # Sort players by points scored
    sorted_players = owner_players.sort_values("Total Points", ascending=False).reset_index(drop=True)

    # Best possible Captain and Vice-Captain based on total points
    best_captain = sorted_players.iloc[0]
    best_vice_captain = sorted_players.iloc[1] if len(sorted_players) > 1 else None

    # Calculate ideal bonus: Captain gets 2x, VC gets 1.5x
    best_captain_bonus = best_captain["Total Points"]
    best_vice_captain_bonus = best_vice_captain["Total Points"] * 0.5 if best_vice_captain is not None else 0

    # Actual total points (without C/VC boost)
    base_points = group["Total Points"].sum()

    # Best possible total
    optimized_total = base_points + best_captain_bonus + best_vice_captain_bonus

    what_if_results.append({
        "Owner": owner,
        "Best Captain": best_captain["Player Name"],
        "Best VC": best_vice_captain["Player Name"] if best_vice_captain is not None else "N/A",
        "Best C/VC Bonus": round(best_captain_bonus + best_vice_captain_bonus),
        "Optimized Team Total": round(optimized_total)
    })

what_if_df = pd.DataFrame(what_if_results).sort_values("Optimized Team Total", ascending=False).reset_index(drop=True)

st.dataframe(
    what_if_df.style.format({
        "Best C/VC Bonus": "{:.0f}",
        "Optimized Team Total": "{:.0f}"
    })
)

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

with st.expander("📋 Last Match Summary"):
    for index, row in df_diff.iterrows():
        owner = row["Owners"]
        gained_points = int(row[latest_col])
        st.write(get_message(gained_points, owner))

# --- Top 4 Appearance Count ---
top4_count = {owner: 0 for owner in df["Owners"]}
for idx, update in enumerate(df.columns[1:]):
    if idx == 0:
        top_gainers = df.nlargest(4, update)
    else:
        top_gainers = df_diff.nlargest(4, update)

    for owner in top_gainers["Owners"]:
        top4_count[owner] += 1

# --- Line Chart Plot ---
st.subheader("📈 Owners Performance Over Time")
fig, ax = plt.subplots(figsize=(15, 5))
cmap = plt.colormaps.get_cmap("tab10")
colors = [cmap(i % 10) for i in range(len(df))]

for i, owner in enumerate(df["Owners"]):
    scores = df[df["Owners"] == owner].values[0][1:]
    ax.plot(df.columns[1:], scores, marker='o', color=colors[i], linewidth=1.2, label=owner)

ax.set_title("Owners Performance Over Updates")
ax.set_xlabel("Updates")
ax.set_ylabel("Total Points")
ax.grid(True)
ax.legend(fontsize=8)
st.pyplot(fig)

# --- Prediction Table ---
st.subheader("📊 Predicted Next Match Scores")

predictions = []
x = np.arange(len(df.columns[1:])).reshape(-1, 1)

for i, owner in enumerate(df["Owners"]):
    y = df.iloc[i, 1:].values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    last_score = y[-1][0]

    # Filter relevant players
    owner_players = points_df[ 
        (points_df["Owner"] == owner) & 
        (points_df["Team"].isin(teams_playing)) &
        (~points_df["Player Name"].isin(non_playing_players))
    ]

    avg_next_points = owner_players["Total Points"].sum() / n_matches_played if not owner_players.empty else 0
    predicted_next = last_score + avg_next_points
    change_pct = ((predicted_next - last_score) / last_score) * 100 if last_score else 0
    top_appearance = top4_count[owner]

    predictions.append([ 
        owner, 
        top_appearance, 
        round(last_score), 
        round(predicted_next), 
        f"{change_pct:.1f}%", 
        owner_players.shape[0]
    ])

merged_df = pd.DataFrame(predictions, columns=[
    "Owners", "Top 4 Appearances", "Last Score", "Predicted Next Score", "Change (%)", "Players in Next Match"
])

merged_df["Projected Final Score"] = merged_df["Last Score"] + \
    (merged_df["Predicted Next Score"] - merged_df["Last Score"]) * (total_matches - n_matches_played)
total_projected = merged_df["Projected Final Score"].sum()
merged_df["Winning Chances (%)"] = (merged_df["Projected Final Score"] / total_projected * 100).round(1)
merged_df.drop(columns=["Projected Final Score"], inplace=True)
merged_df.insert(0, "Rank", merged_df["Winning Chances (%)"].rank(method='first', ascending=False).astype(int))
merged_df = merged_df.sort_values(by="Rank").reset_index(drop=True)

# Display the prediction table
st.dataframe(merged_df, use_container_width=True)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
