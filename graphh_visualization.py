import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stage-4: Visualization", layout="wide")
st.title("📈 Stage-4: Graph Visualization")

df = pd.read_csv("stage3_session_log.csv")

st.subheader("📊 Emotion Distribution")
emotion_summary = df["Emotion"].value_counts(normalize=True) * 100
fig1, ax1 = plt.subplots()
emotion_summary.plot.pie(autopct="%1.1f%%", ax=ax1)
st.pyplot(fig1)

st.subheader("📊 Focus Distribution")
focus_summary = df["Focus"].value_counts(normalize=True) * 100
fig2, ax2 = plt.subplots()
focus_summary.plot.pie(autopct="%1.1f%%", ax=ax2)
st.pyplot(fig2)

st.subheader("📈 Emotion Trend Over Time")
fig3, ax3 = plt.subplots()
ax3.plot(df["Time"], df["Emotion"], marker='o')
st.pyplot(fig3)
