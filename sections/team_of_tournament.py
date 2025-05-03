import streamlit as st
import pandas as pd

def show_team(points_df: pd.DataFrame):
    st.markdown("## 🏆 Team of the Tournament")

    # Exclude Out/Released players
    valid_players_df = points_df[
        ~points_df["Player Name"].str.contains(r"\(O\)|\(R\)|\(RE\)", na=False)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Get top 11 scorers
    top_players = valid_players_df.sort_values(by="Total Points", ascending=False).drop_duplicates(subset=["Player Name"]).head(11).reset_index(drop=True)

    # Assign captain and vice-captain
    if not top_players.empty:
        captain = top_players.loc[0, "Player Name"]
        vice_captain = top_players.loc[1, "Player Name"]

        # Layout for visual structure
        cols = st.columns(3)
        positions = ['Forward', 'Midfielder', 'Defender']  # Visual layout only

        for i, player in top_players.iterrows():
            col = cols[i % 3]
            with col:
                st.markdown(f"### {player['Player Name']}")
                st.markdown(f"**Team:** {player['Team']}")
                st.markdown(f"**Points:** {player['Total Points']}")

                # Highlight captain and VC
                if player["Player Name"] == captain:
                    st.markdown("🧢 **Captain**")
                elif player["Player Name"] == vice_captain:
                    st.markdown("⭐️ **Vice-Captain**")

                st.markdown("---")
    else:
        st.info("Not enough player data available to form the Team of the Tournament.")