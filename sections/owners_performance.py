import streamlit as st
import matplotlib.pyplot as plt

def show_owners_performance(df):
    st.subheader("📈 Owners Performance Over Time", divider="orange")
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
