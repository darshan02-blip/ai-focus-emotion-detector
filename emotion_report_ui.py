import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from datetime import datetime

# ------------------ Page Setup ------------------ #
st.set_page_config(page_title="🧠 AI Emotion Analyzer", layout="wide")

st.title("🧠 AI Emotion Analyzer")

# ------------------ Tabs ------------------ #
tab1, tab2 = st.tabs(["📄 Session Summary", "📊 Live Emotion Dashboard"])

# ------------------ Tab 1: Session Summary ------------------ #
with tab1:
    st.header("📄 Emotion Session Summary Report")

    file_path = "session_report.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()  # Clean column names

        if 'time' not in df.columns or 'emotion' not in df.columns:
            st.error("❌ Required columns ('time' and 'emotion') not found in CSV.")
        elif df.empty:
            st.warning("⚠️ Session data is empty.")
        else:
            df['time'] = pd.to_datetime(df['time'])

            session_start = df['time'].min()
            session_end = df['time'].max()
            session_duration = session_end - session_start

            total_seconds = int(session_duration.total_seconds())
            minutes, seconds = divmod(total_seconds, 60)

            total_frames = len(df)
            emotion_counts = df['emotion'].value_counts(normalize=True) * 100
            dominant_emotion = df['emotion'].value_counts().idxmax()
            top_emotions = emotion_counts.head(3)

            st.markdown("### 🧮 Total Data Points")
            st.markdown(f"`{total_frames}` frames detected")

            st.markdown("### 🕒 Total Session Time")
            st.markdown(f"`{minutes} minutes {seconds} seconds`")

            st.markdown("### 🎭 Dominant Emotion")
            st.markdown(f"`{dominant_emotion}`")

            st.markdown("### 🔝 Top 3 Emotions")
            for emo, pct in top_emotions.items():
                st.markdown(f"- {emo}: `{pct:.2f}%`")

            st.markdown("### 📈 Emotion Distribution (Bar Chart)")
            fig_bar, ax_bar = plt.subplots()
            emotion_counts.plot(kind='bar', ax=ax_bar, color='skyblue')
            ax_bar.set_ylabel("Percentage (%)")
            ax_bar.set_xlabel("Emotions")
            plt.xticks(rotation=45)
            st.pyplot(fig_bar)

            st.markdown("### 🥧 Emotion Distribution (Pie Chart)")
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
            ax_pie.axis('equal')
            st.pyplot(fig_pie)

            # Store dataframe for dashboard tab
            st.session_state.df = df
    else:
        st.warning("⚠️ 'session_report.csv' not found. Please generate a session first.")

# ------------------ Tab 2: Live Emotion Dashboard ------------------ #
with tab2:
    st.header("📊 AI Live Emotion Dashboard")

    emotions = ['happy', 'sad', 'neutral', 'angry', 'surprised', 'fearful', 'disgusted']
    emotion_emojis = {
        'happy': '😄',
        'sad': '😢',
        'neutral': '😐',
        'angry': '😠',
        'surprised': '😲',
        'fearful': '😨',
        'disgusted': '🤢'
    }

    # Simulate live emotion (Replace with real model output in production)
    current_emotion = random.choice(emotions)
    emotion_intensity = random.randint(30, 100)  # Simulated confidence

    st.markdown("### 🔴 Current Emotion")
    st.markdown(
        f"<h1 style='text-align: center; font-size: 72px'>{emotion_emojis[current_emotion]} {current_emotion.upper()}</h1>",
        unsafe_allow_html=True
    )

    st.markdown("### 📊 Emotion Intensity (Simulated)")
    st.progress(emotion_intensity / 100.0)

    # Show bar chart of top 3 from session data if available
    if "df" in st.session_state and "emotion" in st.session_state.df.columns:
        df = st.session_state.df
        emotion_counts = df["emotion"].value_counts(normalize=True) * 100
        top3 = emotion_counts.head(3).round(2).reset_index()
        top3.columns = ["Emotion", "Percentage"]
        top3["Emotion"] = top3["Emotion"].apply(lambda x: f"{emotion_emojis.get(x, '')} {x.capitalize()}")

        st.markdown("### 🏆 Top 3 Emotions (From Session)")
        fig, ax = plt.subplots()
        sns.barplot(x="Percentage", y="Emotion", data=top3, ax=ax, palette="viridis")
        st.pyplot(fig)
    else:
        st.info("📂 No session data available to plot top emotions.")

# ------------------ Footer ------------------ #
st.markdown("---")
st.caption("🎯 Designed with ❤️ using Streamlit & DeepFace")
