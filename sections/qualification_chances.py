# qualification_chances.py
import streamlit as st
import pandas as pd

def show_chances(points_df: pd.DataFrame, match_data: pd.DataFrame, n_matches_played: int):
    st.subheader("⚔️ Owner vs. Owner: Qualification Showdown", divider="orange")
    owners = points_df['Owner'].unique()
    selected_owner1 = st.selectbox("Select Owner 1", owners)
    selected_owner2 = st.selectbox("Select Owner 2", owners, index=1 if len(owners) > 1 else 0)

    if selected_owner1 and selected_owner2 and selected_owner1 != selected_owner2:
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
        if avg_diff > 0:
            st.info(f"On average, :blue[{selected_owner1}] scores :green[{avg_diff:.2f}] more points per match than :red[{selected_owner2}].")
        elif avg_diff < 0:
            st.info(f"On average, :red[{selected_owner2}] scores :green[{abs(avg_diff):.2f}] more points per match than :blue[{selected_owner1}].")
        else:
            st.info("Both owners have a similar average points per match.")

        potential_gain_diff = (max_points_per_match_owner1 - max_points_per_match_owner2) * remaining_matches
        if potential_gain_diff > 0:
            st.warning(f"In the remaining :orange[{remaining_matches}] matches, :blue[{selected_owner1}] has the potential to gain significantly more points if their top players perform exceptionally.")
        elif potential_gain_diff < 0:
            st.warning(f"In the remaining :orange[{remaining_matches}] matches, :red[{selected_owner2}] has the potential to gain significantly more points if their top players perform exceptionally.")
        else:
            st.info("Both owners have similar maximum potential gain in the remaining matches (based on their historical best).")

        realistic_final_diff = (total_points_owner1 + (avg_points_owner1 * remaining_matches)) - (total_points_owner2 + (avg_points_owner2 * remaining_matches))
        if realistic_final_diff > 0:
            st.success(f"Based on current averages, :blue[{selected_owner1}] is projected to finish :green[{realistic_final_diff:.2f}] points ahead of :red[{selected_owner2}].")
        elif realistic_final_diff < 0:
            st.success(f"Based on current averages, :red[{selected_owner2}] is projected to finish :green[{abs(realistic_final_diff):.2f}] points ahead of :blue[{selected_owner1}].")
        else:
            st.info("Based on current averages, both owners are projected to finish with similar total points.")

        st.write("---")
        st.info("Remember, these projections are based on past performance. Strategic captain/vice-captain choices and player form in the upcoming matches will be the deciding factors.")

    elif selected_owner1 == selected_owner2:
        st.warning("Please select two different owners for comparison.")