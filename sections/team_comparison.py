import streamlit as st
import pandas as pd
import plotly.express as px

def show_comparison(points_df: pd.DataFrame, df: pd.DataFrame):
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
        st.dataframe(owner1_df.sort_values("Total Points", ascending=False).head(5)[["Player Name", "Team", "Total Points"]], use_container_width=True, hide_index=True)
    with col4:
        st.markdown(f"**{owner2}'s Top Players**")
        st.dataframe(owner2_df.sort_values("Total Points", ascending=False).head(5)[["Player Name", "Team", "Total Points"]], use_container_width=True, hide_index=True)

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
            max_total_pts = df.iloc[:, -1].max() if not df.empty else 1

            gain = df.loc[df["Owners"] == owner].iloc[:, -1] - df.loc[df["Owners"] == owner].iloc[:, -2]
            max_gain = df.iloc[:, -1].diff().abs().max() if not df.empty else 1

            captain = captain_vc_dict.get(owner, ("", ""))[0]
            vc = captain_vc_dict.get(owner, ("", ""))[1]
            max_player_pts = points_df["Total Points"].max() if not points_df.empty else 1

            captain_pts = points_df.loc[(points_df["Owner"] == owner) & (points_df["Player Name"] == captain), "Total Points"].sum()
            vc_pts = points_df.loc[(points_df["Owner"] == owner) & (points_df["Player Name"] == vc), "Total Points"].sum()

            cvc_total_bonus = points_df.loc[points_df["Owner"] == owner, "CVC Bonus Points"].sum()
            team_total = points_df.loc[points_df["Owner"] == owner, "Total Points"].sum() if not points_df.empty else 1
            max_cvc_bonus_ratio = (points_df["CVC Bonus Points"] / points_df["Total Points"]).max() if not points_df.empty and not (points_df["Total Points"] == 0).any() else 0

            top4 = top4_count.get(owner, 0)
            max_top4 = max(top4_count.values()) if top4_count else 1

            score = (
                (total_pts / max_total_pts) * 10 * 0.3
                + (gain / max_gain if max_gain > 0 else 0) * 10 * 0.15
                + (captain_pts / max_player_pts if max_player_pts > 0 else 0) * 10 * 0.15
                + (vc_pts / max_player_pts if max_player_pts > 0 else 0) * 10 * 0.1
                + (cvc_total_bonus / team_total if team_total > 0 else 0) * 10 * 0.2
                + (top4 / max_top4 if max_top4 > 0 else 0) * 10 * 0.1
            )
            num_stars = round(score / 2)
            if isinstance(num_stars, pd.Series):
                stars = "⭐" * int(max(0, min(5, num_stars.item())))
            else:
                stars = "⭐" * int(max(0, min(5, num_stars)))
            return (round(score, 1)), stars  # Explicitly cast score to float

        score_owner1, stars_owner1 = calculate_owner_rating(owner1)
        rating_rows_comparison.append([owner1, float(score_owner1), stars_owner1]) # Ensure score is a float

        score_owner2, stars_owner2 = calculate_owner_rating(owner2)
        rating_rows_comparison.append([owner2, float(score_owner2), stars_owner2]) # Ensure score is a float

        rating_df_comparison = pd.DataFrame(rating_rows_comparison, columns=["Owner", "Performance Score (out of 10)", "⭐ Rating"])
        st.dataframe(rating_df_comparison, use_container_width=True, hide_index=True)
    else:
        st.warning("Owner rating data is not available for comparison.")