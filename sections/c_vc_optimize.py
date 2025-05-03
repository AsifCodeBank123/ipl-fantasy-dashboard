import streamlit as st
import pandas as pd
import numpy as np

#def show_cvc(points_df):
    # st.subheader("🔮 What-If Best C/VC Optimization",divider="orange")

#     what_if_results = []

#     # Captain and Vice-Captain maps
#     captain_map = {
#         "Mahesh": "Jos Buttler", "Asif": "Pat Cummins", "Pritesh": "Abhishek Sharma",
#         "Pritam": "Suryakumar Yadav", "Lalit": "Shreyas Iyer", "Umesh": "Travis Head",
#         "Sanskar": "Hardik Pandya", "Johnson": "Sunil Naraine", "Somansh": "Rashid Khan",
#         "Wilfred": "Rachin Ravindra"
#     }
#     vice_captain_map = {
#         "Mahesh": "N. Tilak Varma", "Asif": "Venkatesh Iyer", "Pritesh": "Yashasvi Jaiswal",
#         "Pritam": "Virat Kohli", "Lalit": "Shubman Gill", "Umesh": "Rohit Sharma",
#         "Sanskar": "Axar Patel", "Johnson": "Sanju Samson", "Somansh": "Phil Salt",
#         "Wilfred": "KL Rahul"
#     }

#     for owner, group in points_df.groupby("Owner"):
#         owner_players = group.copy()

#         # Base team total (excluding actual C/VC bonus)
#         captain_name = captain_map.get(owner)
#         vice_captain_name = vice_captain_map.get(owner)

#         captain_points = owner_players[owner_players["Player Name"] == captain_name]["Total Points"].sum()
#         vice_captain_points = owner_players[owner_players["Player Name"] == vice_captain_name]["Total Points"].sum()

#         actual_bonus = captain_points + (vice_captain_points * 0.5)
#         actual_total = owner_players["Total Points"].sum()
#         base_total = actual_total - actual_bonus

#         # Best C/VC selection
#         sorted_players = owner_players.sort_values("Total Points", ascending=False).reset_index(drop=True)
#         best_captain = sorted_players.iloc[0]
#         best_vice_captain = sorted_players.iloc[1] if len(sorted_players) > 1 else None

#         best_bonus = best_captain["Total Points"]
#         if best_vice_captain is not None:
#             best_bonus += best_vice_captain["Total Points"] * 0.5

#         optimized_total = base_total + best_bonus

#         what_if_results.append({
#             "Owner": owner,
#             "Best Captain": best_captain["Player Name"],
#             "Best VC": best_vice_captain["Player Name"] if best_vice_captain is not None else "N/A",
#             "Best C/VC Bonus": round(best_bonus),
#             "Optimized Team Total": round(optimized_total)
#         })

#     what_if_df = pd.DataFrame(what_if_results).sort_values("Optimized Team Total", ascending=False).reset_index(drop=True)

#     st.dataframe(
#         what_if_df.style.format({
#             "Best C/VC Bonus": "{:.0f}",
#             "Optimized Team Total": "{:.0f}"
#         })
#     )