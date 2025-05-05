# qualification_chances.py
import streamlit as st
import pandas as pd
import random

def show_chances(points_df: pd.DataFrame, match_data: pd.DataFrame, n_matches_played: int):
    st.subheader("⚔️ Owner vs. Owner: Qualification Showdown", divider="orange")
    owners = points_df['Owner'].unique()
    top_4_owners = points_df.groupby('Owner')['Total Points'].sum().nlargest(4).index.tolist()
    ranked_owners = points_df.groupby('Owner')['Total Points'].sum().sort_values().index.tolist()
    lower_ranked_threshold = ranked_owners[:3]  # Consider bottom 3 as lower ranked

    selected_owner1 = st.selectbox("Select Owner 1", owners)
    selected_owner2 = st.selectbox("Select Owner 2", owners, index=1 if len(owners) > 1 else 0)

    if selected_owner1 and selected_owner2:
        if selected_owner1 == selected_owner2:
            st.warning("Khud se hi competition? Thoda soch leta bhai comparison se pehle! 😉")
        else:
            owner1_df = points_df[points_df['Owner'] == selected_owner1]
            total_points_owner1 = owner1_df['Total Points'].sum()
            avg_points_owner1 = total_points_owner1 / n_matches_played if n_matches_played > 0 else 0
            max_points_per_match_owner1 = owner1_df['Total Points'].max() if not owner1_df.empty else 0
            max_possible_owner1 = total_points_owner1 + (max_points_per_match_owner1 * (14 - n_matches_played))

            owner2_df = points_df[points_df['Owner'] == selected_owner2]
            total_points_owner2 = owner2_df['Total Points'].sum()
            avg_points_owner2 = total_points_owner2 / n_matches_played if n_matches_played > 0 else 0
            max_points_per_match_owner2 = owner2_df['Total Points'].max() if not owner2_df.empty else 0
            max_possible_owner2 = total_points_owner2 + (max_points_per_match_owner2 * (14 - n_matches_played))

            remaining_matches = 14 - n_matches_played

            st.markdown(f"### Comparing :blue[{selected_owner1}] vs. :red[{selected_owner2}]")

            comparison_data = {
                "Owner": [selected_owner1, selected_owner2],
                "Current Points": [f"{total_points_owner1:.2f}", f"{total_points_owner2:.2f}"],
                "Avg. Points/Match": [f"{avg_points_owner1:.2f}", f"{avg_points_owner2:.2f}"],
                "Max Points in a Match": [f"{max_points_per_match_owner1:.2f}", f"{max_points_per_match_owner2:.2f}"],
                "Est. Points in Remaining Matches (Avg.)": [f"{avg_points_owner1 * remaining_matches:.2f}", f"{avg_points_owner2 * remaining_matches:.2f}"],
                "Realistic Final Score": [f"{total_points_owner1 + (avg_points_owner1 * remaining_matches):.2f}",
                                          f"{total_points_owner2 + (avg_points_owner2 * remaining_matches):.2f}"],
                "Maximum Possible Final Score": [f"{max_possible_owner1:.2f}", f"{max_possible_owner2:.2f}"]
            }

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.info("This table provides a comparison based on current performance, average potential, and absolute maximum potential.")

            st.subheader("Detailed Analysis:")

            points_difference = total_points_owner1 - total_points_owner2
            st.write(f"Current Point Difference: :orange[{points_difference:.2f}] ({selected_owner1} {'ahead' if points_difference > 0 else 'behind' if points_difference < 0 else 'tied'}).")

            avg_diff = avg_points_owner1 - avg_points_owner2

            realistic_final_score_owner1 = total_points_owner1 + (avg_points_owner1 * remaining_matches)
            realistic_final_score_owner2 = total_points_owner2 + (avg_points_owner2 * remaining_matches)
            realistic_final_diff = realistic_final_score_owner1 - realistic_final_score_owner2

            max_possible_diff = max_possible_owner1 - max_possible_owner2

            st.subheader("The Road Ahead:")

            if selected_owner1 in top_4_owners and selected_owner2 in top_4_owners:
                top_4_funny_statements = [
                    f" अरे वाह! :blue[{selected_owner1}] और :red[{selected_owner2}] dono top 4 mein! Kya farak padta hai ab, maje karo! 😎",
                    f"Dekho toh! :blue[{selected_owner1}] and :red[{selected_owner2}], top 4 ke VIPs! Chill karo, seat toh booked hai. 😉",
                    f" :blue[{selected_owner1}] vs :red[{selected_owner2}] ka muqabla... top 4 mein ho toh tension kya lena? Enjoy the rest of the league! 😄",
                    f" Kya baat hai! :blue[{selected_owner1}] aur :red[{selected_owner2}]... top 4 club ke members! Ab toh bas position final karni hai, easy peasy! 👍"
                ]
                st.success(random.choice(top_4_funny_statements))
            elif realistic_final_diff > 0:
                st.success(f"Based on current trends, :blue[{selected_owner1}] is on track to finish ahead of :red[{selected_owner2}].")
            elif realistic_final_diff < 0:
                st.success(f"Based on current trends, :red[{selected_owner2}] is on track to finish ahead of :blue[{selected_owner1}].")
            else:
                st.info("Both owners are projected to finish neck and neck!")

            lower_ranker_funny_statements = [
                f":blue[{selected_owner1}], itne points ka farak? Dekh ke compare karna, kahin 'sharm kar le bhai' moment na aa jaaye. 😉",
                f":red[{selected_owner2}], lagta hai thodi zyada mehnat karni padegi. Just saying, comparison soch samajh ke karna! 😂",
                "Hmm, someone's feeling brave with this comparison! Let's see how it plays out. 🤔",
                "Hope this comparison is for motivation, not just for fun! 😅"
            ]

            if (selected_owner1 in lower_ranked_threshold or selected_owner2 in lower_ranked_threshold) and abs(points_difference) > 250 and remaining_matches > 0 and (selected_owner1 not in top_4_owners or selected_owner2 not in top_4_owners):
                st.warning(random.choice(lower_ranker_funny_statements))
            elif remaining_matches == 0:
                st.info("The league stage is over. The final standings are what they are!")

            if points_difference > 500 and avg_diff > 50 and remaining_matches < 5 and (selected_owner1 not in top_4_owners or selected_owner2 not in top_4_owners):
                st.error(f":blue[{selected_owner1}], bhai, :red[{selected_owner2}] se itna farak? Thoda mushkil hai ab. 😅")
            elif points_difference < -500 and avg_diff < -50 and remaining_matches < 5 and (selected_owner1 not in top_4_owners or selected_owner2 not in top_4_owners):
                st.error(f":red[{selected_owner2}], yaar, :blue[{selected_owner1}] bahut aage nikal gaya. Ab toh 'tu rehne de, tere se nahi hoga bhai' wali situation hai. 😉")
            elif points_difference > 1000 and remaining_matches < 3 and (selected_owner1 not in top_4_owners or selected_owner2 not in top_4_owners):
                st.error(f":blue[{selected_owner1}], it's looking like a done deal against :red[{selected_owner2}]! 🎉")
            elif points_difference < -1000 and remaining_matches < 3 and (selected_owner1 not in top_4_owners or selected_owner2 not in top_4_owners):
                st.error(f":red[{selected_owner2}], ab toh :blue[{selected_owner1}] ko pakadna 'Mission Impossible' lag raha hai! 😬")
            else:
                st.info("The remaining matches could still bring about a significant change in fortunes!")

            st.write("---")
            st.info("Remember, these projections are based on past performance. Anything can happen in fantasy cricket!")

    elif selected_owner1 and not selected_owner2:
        st.info("Please select the second owner to start the comparison.")
    elif not selected_owner1 and selected_owner2:
        st.info("Please select the first owner to start the comparison.")