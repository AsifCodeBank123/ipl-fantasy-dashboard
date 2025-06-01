import streamlit as st
import plotly.express as px

def show_owner_insights(points_df):
    st.subheader("📊 Overall Insights", divider="orange")

    role_mapping = {'All': 'Allrounder', 'Bat': 'Batsman', 'Bowl': 'Bowler', 'WK': 'W. Keeper'}
    points_df.loc[:, 'Playing Role'] = points_df['Playing Role'].map(role_mapping).fillna(points_df['Playing Role'])

    st.subheader("Total Points by Owner (Stacked by Playing Role)")
    fig_bar_stacked_owner = px.bar(
        points_df,
        x="Owner",
        y="Total Points",
        color="Playing Role",
        title="Total Points per Owner, Segmented by Playing Role",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    st.plotly_chart(fig_bar_stacked_owner, use_container_width=True)

    st.subheader("🧠 Owner Insights & Breakdown", divider="orange")
    selected_owner = st.selectbox("Select an Owner", sorted(points_df["Owner"].unique()))
    owner_df = points_df[points_df["Owner"] == selected_owner].copy()

    # Calculate team-wise and role-wise breakdowns
    teamwise_df = owner_df.groupby("Team")["Total Points"].sum().reset_index()
    owner_role_points = owner_df.groupby("Playing Role")["Total Points"].sum().reset_index()

    # Add Avg Points column (handle "NA" safely)
    def compute_avg(row):
        try:
            if row["Matches Played"] == "NA":
                return "NA"
            else:
                return round(row["Total Points"] / float(row["Matches Played"]), 1)
        except:
            return "NA"

    owner_df["Avg Points"] = owner_df.apply(compute_avg, axis=1)

    # Display table with new column
    owner_display_df = owner_df[["Player Name", "Team", "Total Points", "Matches Played", "Avg Points", "Player Value", "Playing Role"]]
    owner_display_df = owner_display_df.sort_values(by="Total Points", ascending=False)



    tab1, tab2, tab3 = st.tabs(["Player Contributions (Pie)", "Team Contributions (Bar)", "Role-based Points (Bar)"])

    with tab1:
        fig_pie = px.pie(
            owner_df,
            names="Player Name",
            values="Total Points",
            title=f"{selected_owner}'s Player Contributions",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        fig_bar_team = px.bar(
            teamwise_df,
            x="Team",
            y="Total Points",
            title=f"{selected_owner}'s Team-wise Contributions",
            color="Total Points",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bar_team, use_container_width=True)

    with tab3:
        fig_bar_role = px.bar(
            owner_role_points,
            x="Playing Role",
            y="Total Points",
            title=f"{selected_owner}'s Points by Playing Role",
            color="Playing Role",
            color_discrete_sequence=px.colors.qualitative.Pastel1
        )
        st.plotly_chart(fig_bar_role, use_container_width=True)

    st.markdown(f"#### 📊 Detailed Player Stats for {selected_owner}")
    st.dataframe(owner_display_df, use_container_width=True, hide_index=True)

    top_player = owner_df.sort_values("Total Points", ascending=False).iloc[0]
    bottom_player = owner_df.sort_values("Total Points", ascending=True).iloc[0]

    col3, col4 = st.columns(2)
    with col3:
        st.success(f"🏆 Top Performer: {top_player['Player Name']} ({top_player['Total Points']} pts)")
    with col4:
        st.warning(f"📉 Weakest Performer: {bottom_player['Player Name']} ({bottom_player['Total Points']} pts)")
