import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import LinearRegression

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="IPL Fantasy Dashboard")
st.title("🏏 IPL Fantasy League Performance Dashboard")

# --- Load Data ---
df = pd.read_csv("owners_performance_updates.csv")
points_df = pd.read_csv("points.csv")

# --- Inputs ---
n_matches_played = 5
total_matches = 10

# --- Dropdown of upcoming matches ---
upcoming_matches = [
    "LSG vs GT",
    "SRH vs PBKS",
    "MI vs RR",
    "CSK vs DC",
    "RCB vs KKR"
]

match_input = st.selectbox("Select Upcoming Match", options=upcoming_matches)
teams_playing = match_input.split(" vs ")

# --- Filter players from the selected teams ---
playing_players = points_df[points_df["Team"].isin(teams_playing)]["Player Name"].unique().tolist()
playing_players.sort()

non_playing_players = st.multiselect(
    f"Select Non-Playing Players from {match_input}:",
    options=playing_players,
    default=[]
)

# --- Calculate Update Differences ---
df_diff = df.copy()
df_diff.iloc[:, 2:] = df.iloc[:, 2:].diff(axis=1).fillna(df.iloc[:, 2:3])
latest_col = df_diff.columns[-1]

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

with st.expander("📋 Match Summary Messages"):
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

st.dataframe(merged_df, use_container_width=True)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
