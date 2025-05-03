import streamlit as st
import pandas as pd
import numpy as np

#def mini_auction(points_df):
    # --- Load Unsold Players Data ---
#     try:
#         unsold_df = pd.read_csv("unsold_players.csv")

#         if not unsold_df.empty and "Points" in unsold_df.columns:
#             # Ensure numeric points
#             unsold_df["Points"] = pd.to_numeric(unsold_df["Points"], errors="coerce")
            
#             # Drop rows where points are NaN
#             unsold_df = unsold_df.dropna(subset=["Points"])

#             # Sort and select top performers
#             top_unsold_players = unsold_df.sort_values(by="Points", ascending=False).head(10).reset_index(drop=True)

#             # Display the section
#             st.markdown("## 🔍 Players to Watch Out for in Mini Auction")
#             st.dataframe(top_unsold_players.style.format({"Points": "{:.1f}"}))
#         else:
#             st.warning("Unsold players data is empty or missing 'Points' column.")
#     except FileNotFoundError:
#         st.error("`unsold_players.csv` not found. Please add the file to the project directory.")


#     # --- Trade Suggestions (Advanced with Team Balance Rule) ---

#     st.subheader("🔄💰 Trade Suggestions Based on Team Performance and Budget",divider="orange")

#     # Budget data from your image (could be loaded from a CSV as well)
#     budget_data = {
#         "Mahesh": 40, "Sanskar": 0, "Johnson": 80, "Asif": 310, "Pritam": 40,
#         "Umesh": 30, "Lalit": 0, "Somansh": 0, "Wilfred": 0, "Pritesh": 130
#     }

#     trade_suggestions = []

#     for owner in points_df["Owner"].unique():
#         # Exclude (O) and (S) players locally
#         owner_points = points_df[
#             (points_df["Owner"] == owner) &
#             (~points_df["Player Name"].str.contains(r"\((O|S)\)", regex=True))
#         ].copy()

#         # --- Ensure team formation isn't broken (keep at least 1 player per team) ---
#         team_counts = owner_points["Team"].value_counts()

#         # Find eligible release candidates based on low scores and team balance
#         release_candidates = []
#         for _, row in owner_points.sort_values("Total Points").iterrows():
#             team = row["Team"]
#             player_name = row["Player Name"]
#             if team_counts[team] > 1:
#                 release_candidates.append(row)
#                 team_counts[team] -= 1  # simulate removing this player
#             if len(release_candidates) == 2:
#                 break

#         # If less than 2 can be released (e.g. small team), skip
#         if len(release_candidates) < 2:
#             release_names = [row["Player Name"] for row in release_candidates]
#             value_of_release = 0
#             updated_budget = budget_data.get(owner, 0)
#             trade_suggestions.append({
#                 "Owner": owner,
#                 "Budget Before": budget_data.get(owner, 0),
#                 "Released Players": ", ".join(release_names),
#                 "Value of Released": 0,
#                 "Updated Budget": updated_budget,
#                 "Lowest Scoring Teams": "Not enough eligible releases",
#                 "Suggested Picks": "None"
#             })
#             continue

#         to_release_df = pd.DataFrame(release_candidates)
#         release_names = to_release_df["Player Name"].tolist()

#         # --- Adjust value return based on 200-value rule ---
#         adjusted_values = []
#         for _, row in to_release_df.iterrows():
#             value = row["Player Value"]
#             adjusted = value * 0.5 if value > 200 else value
#             adjusted_values.append(adjusted)

#         value_of_release = sum(adjusted_values)

#         # --- Update budget with adjusted value ---
#         initial_budget = budget_data.get(owner, 0)
#         updated_budget = initial_budget + value_of_release

#         # --- Identify 2 lowest scoring teams for this owner ---
#         team_scores = owner_points.groupby("Team")["Total Points"].sum().sort_values()
#         low_teams = team_scores.head(2).index.tolist()

#         # --- Suggest unsold players from those teams (only if they have >0 points & within budget) ---
#         eligible_unsold = unsold_df[
#             (unsold_df["Team"].isin(low_teams)) &
#             (unsold_df["Points"] > 0) &
#             (unsold_df["Base Price"] <= updated_budget) &
#             (~unsold_df["Player Name"].str.contains(r"\((O|S)\)", regex=True))
#         ].sort_values(by="Points", ascending=False)

#         picks = eligible_unsold.head(2)["Player Name"].tolist()

#         # --- Store suggestion row ---
#         trade_suggestions.append({
#             "Owner": owner,
#             "Budget Before": initial_budget,
#             "Released Players": ", ".join(release_names),
#             "Value of Released": round(value_of_release, 2),
#             "Updated Budget": round(updated_budget, 2),
#             "Lowest Scoring Teams": ", ".join(low_teams),
#             "Suggested Picks": ", ".join(picks) if picks else "None"
#         })

#     trade_df = pd.DataFrame(trade_suggestions)
#     st.dataframe(trade_df, use_container_width=True,hide_index=True)