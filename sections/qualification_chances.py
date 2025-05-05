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
            self_comparison_funny_statements = [
                "Khud ki hi photo dekh raha hai? Confidence ho toh aisa! 😉",
                "Apne pair pe kulhadi maarne ka tareeka thoda casual hai yeh. 😂",
                "Narcissus bhi itna obsessed nahi tha khud se! 😅",
                "Mirror mirror on the wall, who's the fairest of them all? (Spoiler: It's you comparing with yourself!) 😜",
                "Ek hi team, do alag alag angles se analysis... interesting strategy! 🤔",
                "Bhai, akele-akele kya compare kar raha hai? Duniya dekhegi toh kya kahegi! 🤪"
            ]
            st.warning(random.choice(self_comparison_funny_statements))
        else:
            owner1_df = points_df[points_df['Owner'] == selected_owner1]
            total_points_owner1 = owner1_df['Total Points'].sum()
            avg_points_owner1 = total_points_owner1 / n_matches_played if n_matches_played > 0 else 0
            max_points_per_match_owner1 = owner1_df['Total Points'].max() if not owner1_df.empty else 0
            max_possible_owner1 = total_points_owner1 + (max_points_per_match_owner1 * (14 - n_matches_played))
            captain_vc_points_owner1 = owner1_df[owner1_df['CVC Bonus Points'] > 0]['CVC Bonus Points'].sum()

            owner2_df = points_df[points_df['Owner'] == selected_owner2]
            total_points_owner2 = owner2_df['Total Points'].sum()
            avg_points_owner2 = total_points_owner2 / n_matches_played if n_matches_played > 0 else 0
            max_points_per_match_owner2 = owner2_df['Total Points'].max() if not owner2_df.empty else 0
            max_possible_owner2 = total_points_owner2 + (max_points_per_match_owner2 * (14 - n_matches_played))
            captain_vc_points_owner2 = owner2_df[owner2_df['CVC Bonus Points'] > 0]['CVC Bonus Points'].sum()

            remaining_matches = 14 - n_matches_played

            st.markdown(f"### Comparing :blue[{selected_owner1}] vs. :red[{selected_owner2}]")

            comparison_data = {
                "Owner": [selected_owner1, selected_owner2],
                "Current Points": [f"{total_points_owner1:.1f}", f"{total_points_owner2:.1f}"],
                "Avg. Points/Match": [f"{avg_points_owner1:.1f}", f"{avg_points_owner2:.1f}"],
                "Max Points in a Match": [f"{max_points_per_match_owner1:.1f}", f"{max_points_per_match_owner2:.1f}"],
                "Total C/VC Bonus Points": [f"{captain_vc_points_owner1:.1f}", f"{captain_vc_points_owner2:.1f}"],
                "Est. Points in Remaining Matches (Avg.)": [f"{avg_points_owner1 * remaining_matches:.1f}", f"{avg_points_owner2 * remaining_matches:.1f}"],
                "Realistic Final Score": [f"{total_points_owner1 + (avg_points_owner1 * remaining_matches):.1f}",
                                          f"{total_points_owner2 + (avg_points_owner2 * remaining_matches):.1f}"],
                "Maximum Possible Final Score": [f"{max_possible_owner1:.1f}", f"{max_possible_owner2:.1f}"]
            }

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.info("This table provides a detailed comparison.")

            st.subheader("Detailed Analysis:")

            points_difference = total_points_owner1 - total_points_owner2
            st.write(f"Current Point Difference: :orange[{points_difference:.1f}] ({selected_owner1} {'ahead' if points_difference > 0 else 'behind' if points_difference < 0 else 'tied'}).")

            avg_diff = avg_points_owner1 - avg_points_owner2

            realistic_final_score_owner1 = total_points_owner1 + (avg_points_owner1 * remaining_matches)
            realistic_final_score_owner2 = total_points_owner2 + (avg_points_owner2 * remaining_matches)
            realistic_final_diff = realistic_final_score_owner1 - realistic_final_score_owner2

            max_possible_diff = max_possible_owner1 - max_possible_owner2

            st.subheader(f"Path to Victory for :red[{selected_owner2}]")
            if total_points_owner2 < total_points_owner1 and remaining_matches > 0:
                st.markdown(f"Here are a couple of scenarios where :red[{selected_owner2}] could potentially beat :blue[{selected_owner1}]:")
                scenario1_gain = points_difference + 50  # Example: Need to gain current difference + 50
                matches_needed_scenario1 = scenario1_gain / avg_points_owner2 if avg_points_owner2 > 0 else float('inf')
                st.markdown(f"**Scenario 1 (High Scoring Match):** If :blue[{selected_owner1}] scores only their average in the remaining matches, :red[{selected_owner2}] needs to score approximately :orange[{scenario1_gain:.1f}] more points in total across the remaining :orange[{remaining_matches}] matches (about :orange[{scenario1_gain / remaining_matches:.1f}] per match) to overtake. This would require a performance significantly above your average (current avg: :orange[{avg_points_owner2:.1f}]).")

                scenario2_owner1_low = max(0, avg_points_owner1 - 30) # Owner 1 scores below average
                points_to_overtake_scenario2 = (total_points_owner1 + (scenario2_owner1_low * remaining_matches)) - total_points_owner2 + 20 # Overtake by 20
                gain_needed_scenario2 = points_to_overtake_scenario2 / remaining_matches if remaining_matches > 0 else 0
                st.markdown(f"**Scenario 2 (Opponent Underperforms):** If :blue[{selected_owner1}] scores significantly below their average (e.g., around :orange[{scenario2_owner1_low:.1f}] per match), :red[{selected_owner2}] would need to average around :orange[{gain_needed_scenario2:.1f}] points per match in the remaining games to take the lead.")
            elif total_points_owner2 > total_points_owner1:
                st.success(f":red[{selected_owner2}] is currently leading. Maintain the momentum!")
            else:
                st.info("It's a tie! Every point in the remaining matches will be crucial.")

            st.subheader("The Road Ahead:")

            if selected_owner1 in top_4_owners and selected_owner2 in top_4_owners:
                top_4_funny_statements = [
                    f" अरे वाह! :blue[{selected_owner1}] और :red[{selected_owner2}] dono top 4 mein! Ab kya dekhega? Bas maje karo! 😎",
                    f"Top 4 ke sher ho tum dono! :blue[{selected_owner1}] vs :red[{selected_owner2}]... yeh toh bas position ki ladai hai, tension nahi! 😉",
                    f"Lagta hai :blue[{selected_owner1}] aur :red[{selected_owner2}] ko sirf yeh dekhna hai ki number 1 kaun banta hai top 4 mein! 😄",
                    f"Top 4 confirmed! Ab :blue[{selected_owner1}] aur :red[{selected_owner2}] bas yeh decide kar rahe hain ki podium pe left se jaana hai ya right se! 👍",
                    f"Kya baat hai! :blue[{selected_owner1}] aur :red[{selected_owner2}]... qualification toh jeb mein hai! Ab dekhte hain ego kitna hurt hota hai haarne pe! 😜",
                    f"Dono top 4 mein? Yeh comparison toh 'baap baap hota hai' ya 'haathi ke daant khane ke aur, dikhane ke aur' wala scene hai! 😂"
                ]
                st.success(random.choice(top_4_funny_statements))
            elif realistic_final_diff > 0:
                st.success(f"Based on current trends, :blue[{selected_owner1}] is on track to finish ahead of :red[{selected_owner2}].")
            elif realistic_final_diff < 0:
                st.success(f"Based on current trends, :red[{selected_owner2}] is on track to finish ahead of :blue[{selected_owner1}].")
            else:
                st.info("Both owners are projected to finish neck and neck!")

            lower_ranker_funny_statements = [
                f":blue[{selected_owner1}], itne points ka farak? Dekh ke compare karna, kahin 'sharm kar le bhai' moment na aa jaaye! 😬",
                f":red[{selected_owner2}], lagta hai thodi zyada 'miracle' ki zaroorat hai. Comparison toh theek hai, par reality check bhi zaroori hai! 😂",
                f"Someone's comparing the Everest with a molehill? Just kidding... mostly! 😉 (:blue[{selected_owner1}] or :red[{selected_owner2}], guess who!)",
                f"Hope this comparison is a source of motivation, aur 'arre yaar, kahan phas gaya' wali feeling nahi aa rahi hogi! 😅 (:blue[{selected_owner1}] or :red[{selected_owner2}], handle with care!)",
                f":blue[{selected_owner1}] aur :red[{selected_owner2}] ka comparison... ek taraf 'jeet ka jhanda', doosri taraf 'koshish jaari hai' wala vibe! 🤔",
                f"Bhai, agar zyada hi farak hai toh comparison se pehle ek baar leaderboard dekh lena chahiye tha! Just saying! 🤪 (:blue[{selected_owner1}] or :red[{selected_owner2}], no offense!)"
            ]

            if (selected_owner1 in lower_ranked_threshold or selected_owner2 in lower_ranked_threshold) and abs(points_difference) > 250 and remaining_matches > 0 and (selected_owner1 not in top_4_owners or selected_owner2 not in top_4_owners):
                st.warning(random.choice(lower_ranker_funny_statements))
            elif remaining_matches == 0:
                st.info("The league stage is over. The final standings are what they are!")

            if points_difference > 500 and avg_diff > 50 and remaining_matches < 5 and (selected_owner1 not in top_4_owners or selected_owner2 not in top_4_owners):
                st.error(f":blue[{selected_owner1}], bhai, :red[{selected_owner2}] se itna farak? Ab toh 'thoda mushkil hai' understatement hoga! 😅")
            elif points_difference < -500 and avg_diff < -50 and remaining_matches < 5 and (selected_owner1 not in top_4_owners or selected_owner2 not in top_4_owners):
                st.error(f":red[{selected_owner2}], yaar, :blue[{selected_owner1}] bahut aage nikal gaya. Ab toh 'tu rehne de, tere se nahi hoga bhai' is the official anthem! 😉")
            elif points_difference > 1000 and remaining_matches < 3 and (selected_owner1 not in top_4_owners or selected_owner2 not in top_4_owners):
                st.error(f":blue[{selected_owner1}], yeh toh 'ek tarfa pyaar' wala comparison ho gaya! 🎉 :red[{selected_owner2}], next season try karna!")
            elif points_difference < -1000 and remaining_matches < 3 and (selected_owner1 not in top_4_owners or selected_owner2 not in top_4_owners):
                st.error(f":red[{selected_owner2}], ab toh :blue[{selected_owner1}] ko pakadna 'bhagwan bharose' wali baat hai! 😬 All the best for the next match... maybe!")
            else:
                st.info("The remaining matches could still bring about a significant change in fortunes!")

            st.write("---")
            st.info("Remember, these projections are based on past performance. Anything can happen in fantasy cricket!")

    elif selected_owner1 and not selected_owner2:
        st.info("Please select the second owner to start the comparison.")
    elif not selected_owner1 and selected_owner2:
        st.info("Please select the first owner to start the comparison.")