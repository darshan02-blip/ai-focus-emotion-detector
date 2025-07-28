import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="📜 Session History", layout="wide")
st.title("📜 AI Focus & Emotion Detector - Session History")

# ✅ Create reports folder if not exists
os.makedirs("reports", exist_ok=True)

# ✅ Detect existing users
history_files = [f for f in os.listdir("reports") if f.startswith("history_") and f.endswith(".csv")]
users = [f.replace("history_", "").replace(".csv", "") for f in history_files]

# ✅ Select or create user
username = st.selectbox("Select User", options=["-- New User --"] + users)

if username == "-- New User --":
    new_name = st.text_input("Enter new username:")
    if st.button("Create User") and new_name.strip():
        username = new_name.strip()
        st.success(f"✅ User '{username}' created!")
    else:
        st.stop()

# ✅ History file for this user
history_file = f"reports/history_{username}.csv"

if os.path.exists(history_file):
    df = pd.read_csv(history_file)
else:
    df = pd.DataFrame(columns=["Session Start", "Session End", "Duration", "Emotions", "Focus", "PDF Report"])

# ✅ Display Session Table
st.subheader(f"📊 Session History for {username}")
if df.empty:
    st.info("No sessions found yet for this user.")
else:
    st.dataframe(df)

    # ✅ Aggregate Emotion and Focus Distributions
    all_emotions = []
    all_focus = []

    for e in df["Emotions"]:
        all_emotions.extend([x.strip() for x in e.split(",")])
    for f in df["Focus"]:
        all_focus.extend([x.strip() for x in f.split(",")])

    if all_emotions:
        st.subheader("📊 Average Emotion Distribution")
        emotion_counts = pd.Series(all_emotions).value_counts(normalize=True) * 100
        fig1, ax1 = plt.subplots()
        emotion_counts.plot.pie(autopct="%1.1f%%", ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

    if all_focus:
        st.subheader("📊 Average Focus Distribution")
        focus_counts = pd.Series(all_focus).value_counts(normalize=True) * 100
        fig2, ax2 = plt.subplots()
        focus_counts.plot.pie(autopct="%1.1f%%", ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2)

    # ✅ Session Duration Trend
    if "Duration" in df.columns and not df["Duration"].empty:
        st.subheader("⏱️ Session Duration Over Time")
        duration_minutes = df["Duration"].apply(lambda x: int(x.split('m')[0]) + int(x.split('m')[1].replace('s', ''))/60)
        fig3, ax3 = plt.subplots()
        ax3.plot(df["Session Start"], duration_minutes, marker='o')
        ax3.set_xlabel("Session Date")
        ax3.set_ylabel("Duration (minutes)")
        ax3.set_title("Session Duration Trend")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

    # ✅ Download Past PDFs
    st.subheader("⬇️ Download Past PDF Reports")
    for idx, row in df.iterrows():
        if "PDF Report" in row and isinstance(row["PDF Report"], str) and os.path.exists(row["PDF Report"]):
            with open(row["PDF Report"], "rb") as file:
                st.download_button(f"Download Report - {row['Session Start']}", file, file_name=os.path.basename(row["PDF Report"]))
