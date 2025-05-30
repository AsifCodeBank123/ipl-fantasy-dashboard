import pandas as pd
import streamlit as st

def clean_name(name):
    return name.replace('[C]', '').replace('[VC]', '').strip()

def show_awards(points_df):
    st.subheader("🏅 Best/Worst Player Awards")

    # Clean role mapping
    points_df['Role'] = points_df['Playing Role'].map({
        'WK': 'Wicketkeeper',
        'Opener': 'Batter',
        'Bat': 'Batter',
        'Bowl': 'Bowler',
        'All': 'Allrounder'
    })

    # Identify best and worst by role
    best_by_role = points_df.loc[points_df.groupby('Role')['Total Points'].idxmax()]
    worst_by_role = points_df.loc[points_df.groupby('Role')['Total Points'].idxmin()]

    # Captain & VC logic
    points_df['Is_Captain'] = points_df['Player Name'].str.contains(r'\[C\]')
    points_df['Is_VC'] = points_df['Player Name'].str.contains(r'\[VC\]')

    best_captain = points_df[points_df['Is_Captain']].sort_values('Total Points', ascending=False).iloc[0]
    worst_captain = points_df[points_df['Is_Captain']].sort_values('Total Points', ascending=True).iloc[0]

    best_vc = points_df[points_df['Is_VC']].sort_values('Total Points', ascending=False).iloc[0]
    worst_vc = points_df[points_df['Is_VC']].sort_values('Total Points', ascending=True).iloc[0]

    # Best and worst C/VC combo (same owner)
    owners = points_df['Owner'].unique()
    combo_data = []

    for owner in owners:
        owner_df = points_df[points_df['Owner'] == owner]
        capt = owner_df[owner_df['Is_Captain']]
        vc = owner_df[owner_df['Is_VC']]
        if not capt.empty and not vc.empty:
            total = capt.iloc[0]['Total Points'] + vc.iloc[0]['Total Points']
            combo_data.append((owner, capt.iloc[0]['Player Name'], vc.iloc[0]['Player Name'], total))

    best_combo = max(combo_data, key=lambda x: x[3])
    worst_combo = min(combo_data, key=lambda x: x[3])

    # Trades
    best_trade = ("Abhishek Porel", points_df[points_df["Player Name"].str.contains("Abhishek Porel", case=False)]["Total Points"].values[0], "Lalit")
    worst_trade = ("Tushar Deshpande", points_df[points_df["Player Name"].str.contains("Tushar Deshpande", case=False)]["Total Points"].values[0], "Asif")

    # Compile award list
    awards = [
        ["Best Captain", clean_name(best_captain['Player Name']), best_captain['Total Points'], best_captain['Owner']],
        ["Worst Captain", clean_name(worst_captain['Player Name']), worst_captain['Total Points'], worst_captain['Owner']],
        ["Best Vice-Captain", clean_name(best_vc['Player Name']), best_vc['Total Points'], best_vc['Owner']],
        ["Worst Vice-Captain", clean_name(worst_vc['Player Name']), worst_vc['Total Points'], worst_vc['Owner']],
        ["Best C/VC Combo", f"{clean_name(best_combo[1])} + {clean_name(best_combo[2])}", best_combo[3], best_combo[0]],
        ["Worst C/VC Combo", f"{clean_name(worst_combo[1])} + {clean_name(worst_combo[2])}", worst_combo[3], worst_combo[0]],
    ]

    # Add best/worst by role
    for role in ['Batter', 'Bowler', 'Wicketkeeper', 'Allrounder']:
        best = best_by_role[best_by_role['Role'] == role].iloc[0]
        worst = worst_by_role[worst_by_role['Role'] == role].iloc[0]
        awards.append([f"Best {role}", clean_name(best['Player Name']), best['Total Points'], best['Owner']])
        awards.append([f"Worst {role}", clean_name(worst['Player Name']), worst['Total Points'], worst['Owner']])

    # Add trades
    awards.append(["Best Trade", clean_name(best_trade[0]), best_trade[1], best_trade[2]])
    awards.append(["Worst Trade", clean_name(worst_trade[0]), worst_trade[1], worst_trade[2]])

    awards_df = pd.DataFrame(awards, columns=["Category", "Player Name", "Points", "Owner"])
    st.dataframe(awards_df, use_container_width=True, hide_index=True)
