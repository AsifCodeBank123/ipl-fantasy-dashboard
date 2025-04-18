import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import pytz
import random
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="HPL Fantasy Dashboard")
st.header("🏏 :orange[HPL] Fantasy League Performance Dashboard", divider = "orange")

# --- Inputs ---
n_matches_played = 6
total_matches = 14

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


match_data = pd.read_csv("match_schedule.csv")
# Convert match data into a DataFrame for easier manipulation
match_df = pd.DataFrame(match_data, columns=["Date", "Time", "Match"])

# --- Load Data ---
df = pd.read_csv("owners_performance_updates.csv")
points_df = pd.read_csv("points.csv")

# Optional: wrap column headers or shorten names in your DataFrame
points_df.columns = [col if len(col) < 15 else col[:12] + "..." for col in points_df.columns]

# Define IST timezone
ist = pytz.timezone("Asia/Kolkata")
# --- Get current time ---
current_time = datetime.now(ist)

# Convert match date & time to timezone-aware datetime
# Combine Date and Time into a single DateTime column
match_df['DateTime'] = match_df['Date'] + ' ' + match_df['Time']
# Convert to datetime
match_df['DateTime'] = pd.to_datetime(match_df['DateTime'], format='%d-%b-%y %I:%M %p').dt.tz_localize("Asia/Kolkata")

# Filter upcoming matches
upcoming_matches_df = match_df[match_df["DateTime"] > current_time].copy()
upcoming_matches_df["TimeDiffInHours"] = (upcoming_matches_df["DateTime"] - current_time).dt.total_seconds() / 3600

# Select next match
if not upcoming_matches_df.empty:
    next_match_row = upcoming_matches_df[upcoming_matches_df["TimeDiffInHours"] <= 4].head(1)
    next_match = next_match_row["Match"].values[0] if not next_match_row.empty else upcoming_matches_df.iloc[0]["Match"]
else:
    next_match = "No upcoming match"



# --- Dropdown of upcoming matches ---
upcoming_matches = upcoming_matches_df["Match"].tolist()

# Dropdown to select match (pre-select next match if found)
match_input = st.selectbox("Current/Next Match", options=upcoming_matches_df["Match"].tolist(), index=0)
if match_input:
    # Split the input and strip whitespace
    teams_playing = [team.strip() for team in match_input.split("vs")]
    
    if len(teams_playing) == 2:
        team1, team2 = teams_playing
        # Get unique player names for the selected teams
        match_players = sorted(points_df[points_df["Team"].isin([team1, team2])]["Player Name"].unique())
        
        # Create a multiselect for non-playing players
        non_playing_players = st.multiselect(
            f"Select Non-Playing Players from {team1}/{team2}:",
            options=match_players
        )
    else:
        st.error("Match format error. Please check the selected match.")
else:
    st.error("No match selected.")


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

# Sidebar navigation
with st.sidebar.expander("📂 Select Section", expanded=True):
    section=st.radio("",[
        "Owner Rankings: Current vs Predicted",
        "Player Impact - Next Match Focus",
        "Captain/Vice-Captain Impact Analysis",
        "Best C/VC Suggestion",
        "Players to Watch Out for in Mini Auction",
        "Owner Insights & Breakdown",
        "Owners Performance"
    ])


if section == "Owner Rankings: Current vs Predicted":
    # --- Prediction Table ---
    st.subheader("📊🏆 Owner Rankings: Current vs Predicted (Next Match Impact)",divider="orange")

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
            (last_score),
            int(predicted_next),
            f"{change_pct:.1f}%",
            owner_players.shape[0],
            top_appearance
        ])

    merged_df = pd.DataFrame(predictions, columns=[
        "Owners", "Current Score", "Predicted Next Score",
        "Change (%)", "Players in Next Match","Top 4 Appearances"
    ])

    # --- Add Rank Delta Columns ---
    merged_df = merged_df.sort_values(by="Current Score", ascending=False).reset_index(drop=True)
    last_scores = merged_df["Current Score"].values

    def format_delta(delta, decimals=1):
        """Format delta to show integer if no decimal part, else round to the specified decimal places."""
        if delta.is_integer():  # If the delta is an integer, show as integer
            return int(delta)
        else:
            return round(delta, decimals)  # Else, show rounded to the specified decimal places


    next_rank_deltas = []
    first_rank_deltas = []

    for idx, score in enumerate(last_scores):
        # Delta to next higher rank (i.e., previous in sorted list)
        if idx == 0:
            next_rank_deltas.append("")
            first_rank_deltas.append("")
        else:
            next_rank_deltas.append(format_delta(last_scores[idx - 1] - score, 1))
            first_rank_deltas.append(format_delta(last_scores[0] - score, 1))

    # Insert delta columns after Last Score
    merged_df.insert(2, "Next Rank Delta", next_rank_deltas)
    merged_df.insert(3, "1st Rank Delta", first_rank_deltas)

    # Winning Chances
    merged_df["Projected Final Score"] = merged_df["Current Score"] + \
        (merged_df["Predicted Next Score"] - merged_df["Current Score"]) * (total_matches - n_matches_played)
    total_projected = merged_df["Projected Final Score"].sum()
    merged_df["Winning Chances (%)"] = (merged_df["Projected Final Score"] / total_projected * 100).round(1)
    merged_df.drop(columns=["Projected Final Score"], inplace=True)

    # Rank based on Last Score
    merged_df.insert(0, "Rank", merged_df["Current Score"].rank(method='first', ascending=False).astype(int))
    merged_df = merged_df.sort_values(by="Rank").reset_index(drop=True)

    # Center-align the DataFrame
    st.markdown(
        f"<style>div.stDataFrame {{text-align: center;}}</style>",
        unsafe_allow_html=True
    )

    # Display the prediction table
    st.dataframe(merged_df, use_container_width=True, hide_index=True)


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
    # --- Player Impact Table ---
    st.subheader("🧠 Player Impact - Next Match Focus",divider="orange")
    impact_df = points_df[(points_df["Team"].isin(teams_playing)) & (~points_df["Player Name"].isin(non_playing_players))]
    top_players = impact_df.sort_values(by="Total Points", ascending=False).head(10)
    st.dataframe(top_players[["Player Name", "Team", "Owner", "Total Points"]], use_container_width=True,hide_index=True)

    
elif section == "Captain/Vice-Captain Impact Analysis":
   # Group by owner
    owner_cvc_summary = points_df.groupby("Owner").agg(
        Team_Total_Points=("Total Points", "sum"),
        CVC_Bonus_Points=("CVC Bonus Points", "sum")
    ).reset_index()

    owner_cvc_summary["CVC_Impact_%"] = (owner_cvc_summary["CVC_Bonus_Points"] / owner_cvc_summary["Team_Total_Points"]) * 100
    owner_cvc_summary = owner_cvc_summary.sort_values(by="CVC_Bonus_Points", ascending=False)

   
    st.subheader("💥 Captain/Vice-Captain Impact Analysis",divider="orange")
    st.dataframe(owner_cvc_summary.style.format({"CVC_Impact_%": "{:.0f}%", "CVC_Bonus_Points": "{:.0f}", "Team_Total_Points": "{:.0f}"}))

elif section == "Best C/VC Suggestion":
    st.subheader("🔮 What-If Best C/VC Optimization",divider="orange")

    what_if_results = []

    # Captain and Vice-Captain maps
    captain_map = {
        "Mahesh": "Jos Buttler", "Asif": "Pat Cummins", "Pritesh": "Abhishek Sharma",
        "Pritam": "Suryakumar Yadav", "Lalit": "Shreyas Iyer", "Umesh": "Travis Head",
        "Sanskar": "Hardik Pandya", "Johnson": "Sunil Naraine", "Somansh": "Rashid Khan",
        "Wilfred": "Rachin Ravindra"
    }
    vice_captain_map = {
        "Mahesh": "N. Tilak Varma", "Asif": "Venkatesh Iyer", "Pritesh": "Yashasvi Jaiswal",
        "Pritam": "Virat Kohli", "Lalit": "Shubman Gill", "Umesh": "Rohit Sharma",
        "Sanskar": "Axar Patel", "Johnson": "Sanju Samson", "Somansh": "Phil Salt",
        "Wilfred": "KL Rahul"
    }

    for owner, group in points_df.groupby("Owner"):
        owner_players = group.copy()

        # Base team total (excluding actual C/VC bonus)
        captain_name = captain_map.get(owner)
        vice_captain_name = vice_captain_map.get(owner)

        captain_points = owner_players[owner_players["Player Name"] == captain_name]["Total Points"].sum()
        vice_captain_points = owner_players[owner_players["Player Name"] == vice_captain_name]["Total Points"].sum()

        actual_bonus = captain_points + (vice_captain_points * 0.5)
        actual_total = owner_players["Total Points"].sum()
        base_total = actual_total - actual_bonus

        # Best C/VC selection
        sorted_players = owner_players.sort_values("Total Points", ascending=False).reset_index(drop=True)
        best_captain = sorted_players.iloc[0]
        best_vice_captain = sorted_players.iloc[1] if len(sorted_players) > 1 else None

        best_bonus = best_captain["Total Points"]
        if best_vice_captain is not None:
            best_bonus += best_vice_captain["Total Points"] * 0.5

        optimized_total = base_total + best_bonus

        what_if_results.append({
            "Owner": owner,
            "Best Captain": best_captain["Player Name"],
            "Best VC": best_vice_captain["Player Name"] if best_vice_captain is not None else "N/A",
            "Best C/VC Bonus": round(best_bonus),
            "Optimized Team Total": round(optimized_total)
        })

    what_if_df = pd.DataFrame(what_if_results).sort_values("Optimized Team Total", ascending=False).reset_index(drop=True)

    st.dataframe(
        what_if_df.style.format({
            "Best C/VC Bonus": "{:.0f}",
            "Optimized Team Total": "{:.0f}"
        })
    )


elif section == "Players to Watch Out for in Mini Auction":
    # --- Load Unsold Players Data ---
    try:
        unsold_df = pd.read_csv("unsold_players.csv")

        if not unsold_df.empty and "Points" in unsold_df.columns:
            # Ensure numeric points
            unsold_df["Points"] = pd.to_numeric(unsold_df["Points"], errors="coerce")
            
            # Drop rows where points are NaN
            unsold_df = unsold_df.dropna(subset=["Points"])

            # Sort and select top performers
            top_unsold_players = unsold_df.sort_values(by="Points", ascending=False).head(10).reset_index(drop=True)

            # Display the section
            st.markdown("## 🔍 Players to Watch Out for in Mini Auction")
            st.dataframe(top_unsold_players.style.format({"Points": "{:.1f}"}))
        else:
            st.warning("Unsold players data is empty or missing 'Points' column.")
    except FileNotFoundError:
        st.error("`unsold_players.csv` not found. Please add the file to the project directory.")


    # --- Trade Suggestions (Advanced with Team Balance Rule) ---

    st.subheader("🔄💰 Trade Suggestions Based on Team Performance and Budget",divider="orange")

    # Budget data from your image (could be loaded from a CSV as well)
    budget_data = {
        "Mahesh": 40, "Sanskar": 0, "Johnson": 80, "Asif": 310, "Pritam": 40,
        "Umesh": 30, "Lalit": 0, "Somansh": 0, "Wilfred": 0, "Pritesh": 130
    }

    trade_suggestions = []

    for owner in points_df["Owner"].unique():
        # Exclude (O) and (S) players locally
        owner_points = points_df[
            (points_df["Owner"] == owner) &
            (~points_df["Player Name"].str.contains(r"\((O|S)\)", regex=True))
        ].copy()

        # --- Ensure team formation isn't broken (keep at least 1 player per team) ---
        team_counts = owner_points["Team"].value_counts()

        # Find eligible release candidates based on low scores and team balance
        release_candidates = []
        for _, row in owner_points.sort_values("Total Points").iterrows():
            team = row["Team"]
            player_name = row["Player Name"]
            if team_counts[team] > 1:
                release_candidates.append(row)
                team_counts[team] -= 1  # simulate removing this player
            if len(release_candidates) == 2:
                break

        # If less than 2 can be released (e.g. small team), skip
        if len(release_candidates) < 2:
            release_names = [row["Player Name"] for row in release_candidates]
            value_of_release = 0
            updated_budget = budget_data.get(owner, 0)
            trade_suggestions.append({
                "Owner": owner,
                "Budget Before": budget_data.get(owner, 0),
                "Released Players": ", ".join(release_names),
                "Value of Released": 0,
                "Updated Budget": updated_budget,
                "Lowest Scoring Teams": "Not enough eligible releases",
                "Suggested Picks": "None"
            })
            continue

        to_release_df = pd.DataFrame(release_candidates)
        release_names = to_release_df["Player Name"].tolist()

        # --- Adjust value return based on 200-value rule ---
        adjusted_values = []
        for _, row in to_release_df.iterrows():
            value = row["Player Value"]
            adjusted = value * 0.5 if value > 200 else value
            adjusted_values.append(adjusted)

        value_of_release = sum(adjusted_values)

        # --- Update budget with adjusted value ---
        initial_budget = budget_data.get(owner, 0)
        updated_budget = initial_budget + value_of_release

        # --- Identify 2 lowest scoring teams for this owner ---
        team_scores = owner_points.groupby("Team")["Total Points"].sum().sort_values()
        low_teams = team_scores.head(2).index.tolist()

        # --- Suggest unsold players from those teams (only if they have >0 points & within budget) ---
        eligible_unsold = unsold_df[
            (unsold_df["Team"].isin(low_teams)) &
            (unsold_df["Points"] > 0) &
            (unsold_df["Base Price"] <= updated_budget) &
            (~unsold_df["Player Name"].str.contains(r"\((O|S)\)", regex=True))
        ].sort_values(by="Points", ascending=False)

        picks = eligible_unsold.head(2)["Player Name"].tolist()

        # --- Store suggestion row ---
        trade_suggestions.append({
            "Owner": owner,
            "Budget Before": initial_budget,
            "Released Players": ", ".join(release_names),
            "Value of Released": round(value_of_release, 2),
            "Updated Budget": round(updated_budget, 2),
            "Lowest Scoring Teams": ", ".join(low_teams),
            "Suggested Picks": ", ".join(picks) if picks else "None"
        })

    trade_df = pd.DataFrame(trade_suggestions)
    st.dataframe(trade_df, use_container_width=True,hide_index=True)

elif section == "Owner Insights & Breakdown":
    # --- Owner Insights Block ---

    st.subheader("🧠 Owner Insights & Breakdown",divider="orange")

    selected_owner = st.selectbox("Select an Owner", sorted(points_df["Owner"].unique()))

    owner_df = points_df[points_df["Owner"] == selected_owner]

    # Prepare data
    teamwise_df = owner_df.groupby("Team")["Total Points"].sum().reset_index()
    owner_display_df = owner_df[["Player Name", "Team", "Total Points", "Player Value"]].sort_values(by="Total Points", ascending=False)

    # Layout for side-by-side charts
    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            owner_df,
            names="Player Name",
            values="Total Points",
            title=f"{selected_owner}'s Player Contributions",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            teamwise_df,
            x="Team",
            y="Total Points",
            title=f"{selected_owner}'s Team-wise Contributions",
            color="Total Points",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Player table
    st.markdown(f"#### 📊 Detailed Player Stats for {selected_owner}")
    st.dataframe(owner_display_df, use_container_width=True,hide_index=True)

    # Top & Bottom Performer
    top_player = owner_df.sort_values("Total Points", ascending=False).iloc[0]
    bottom_player = owner_df.sort_values("Total Points", ascending=True).iloc[0]

    col3, col4 = st.columns(2)
    with col3:
        st.success(f"🏆 Top Performer: {top_player['Player Name']} ({top_player['Total Points']} pts)")
    with col4:
        st.warning(f"📉 Weakest Performer: {bottom_player['Player Name']} ({bottom_player['Total Points']} pts)")

elif section == "Owners Performance":
    # --- Line Chart Plot ---
    st.subheader("📈 Owners Performance Over Time",divider="orange")
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


st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
