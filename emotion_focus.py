import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
from datetime import datetime

# ========== FOLDERS ==========
os.makedirs("assets", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="🧠 AI Focus & Emotion Detector", layout="wide")
# === Custom CSS for Modern UI ===
st.markdown("""
    <style>
        body {
            background-color: #0f172a;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 34px;
            font-weight: bold;
            color: #f3f4f6;
            margin-bottom: 20px;
        }
        .user-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            margin: auto;
            max-width: 600px;
        }
        .session-info {
            font-size: 14px;
            color: #e5e7eb;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🧠 AI Focus & Emotion Detection</div>", unsafe_allow_html=True)

# ========== SESSION STATE ==========
if "run" not in st.session_state:
    st.session_state.run = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []
if "focus_log" not in st.session_state:
    st.session_state.focus_log = []
if "time_log" not in st.session_state:
    st.session_state.time_log = []
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# === User Management ===
users_file = "user_history.csv"
if not os.path.exists(users_file):
    pd.DataFrame(columns=["Name", "Last Session", "Duration"]).to_csv(users_file, index=False)

user_df = pd.read_csv(users_file)
existing_users = user_df["Name"].unique().tolist() if not user_df.empty else []

st.markdown("<div class='user-card'>", unsafe_allow_html=True)
selected_user = st.selectbox("Select Your Name", ["New User"] + existing_users)

if selected_user == "New User":
    name_input = st.text_input("Enter your name:")
    if name_input:
        st.session_state.user_name = name_input
else:
    st.session_state.user_name = selected_user
    # Show last session info
    last = user_df[user_df["Name"] == selected_user].tail(1)
    if not last.empty:
        last_date = last["Last Session"].values[0]
        last_dur = last["Duration"].values[0]
        st.markdown(f"<div class='session-info'>👋 Welcome back <b>{selected_user}</b>!<br>Last session: {last_date}<br>Duration: {last_dur}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ========== CONTROLS ==========
fps = st.slider("Frame Rate (FPS)", 1, 15, 5)
resolution = st.selectbox("Camera Resolution", ["480p", "720p"])

frame_placeholder = st.empty()
timer_placeholder = st.empty()

# ========== START SESSION ==========
if st.button("▶️ Start Session") and not st.session_state.run and st.session_state.user_name:
    st.session_state.run = True
    st.session_state.start_time = datetime.now()
    st.session_state.emotion_log.clear()
    st.session_state.focus_log.clear()
    st.session_state.time_log.clear()

# ========== STOP SESSION ==========
if st.session_state.run:
    if st.button("⏹️ Stop Session"):
        st.session_state.run = False

# ========== LIVE CAMERA ==========
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    if resolution == "480p":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        res_label = "480p"
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        res_label = "720p"

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Live timer
        elapsed = datetime.now() - st.session_state.start_time
        mins, secs = divmod(int(elapsed.total_seconds()), 60)
        live_timer = f"{mins:02}:{secs:02}"
        timer_placeholder.subheader(f"⏱️ Time: {live_timer} | 🎥 FPS: {fps} | 📷 Resolution: {res_label}")

        # Emotion detection
        try:
            analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotions = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
            dominant_emotion = max(emotions, key=emotions.get)
        except:
            dominant_emotion = "unknown"

        st.session_state.emotion_log.append(dominant_emotion)
        st.session_state.time_log.append(int(elapsed.total_seconds()))

        # Focus detection
        focus = "Focused"
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            def ear(eye):
                A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
                B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
                C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
                return (A + B) / (2.0 * C)

            left_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [face_landmarks.landmark[i] for i in [263, 387, 385, 362, 380, 373]]
            avg_ear = (ear(left_eye) + ear(right_eye)) / 2.0
            if avg_ear < 0.18:
                focus = "Not Focused"

        st.session_state.focus_log.append(focus)

        # Display frame
        cv2.putText(frame, f"Time: {live_timer}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps} | Res: {res_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Focus: {focus}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        time.sleep(1 / fps)

    cap.release()

# ========== RESULTS ==========
if not st.session_state.run and st.session_state.start_time and st.session_state.emotion_log:
    end_time = datetime.now()
    start_time = st.session_state.start_time
    duration_seconds = (end_time - start_time).total_seconds()
    minutes, seconds = divmod(duration_seconds, 60)
    duration = f"{int(minutes)}m {int(seconds)}s"

    df = pd.DataFrame({
        "Time": st.session_state.time_log,
        "Emotion": st.session_state.emotion_log,
        "Focus": st.session_state.focus_log
    })
    df.to_csv("emotion_log.csv", index=False)

    # Pie Charts
    st.subheader("📊 Emotion & Focus Distribution")
    emotion_summary = df["Emotion"].value_counts(normalize=True) * 100
    focus_summary = df["Focus"].value_counts(normalize=True) * 100

    fig1, ax1 = plt.subplots()
    if not emotion_summary.empty:
        emotion_summary.plot.pie(autopct="%1.1f%%", ax=ax1)
    ax1.set_title("Emotion Distribution")
    ax1.set_ylabel("")
    plt.savefig("assets/emotion_pie.png")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    if not focus_summary.empty:
        focus_summary.plot.pie(autopct="%1.1f%%", ax=ax2)
    ax2.set_title("Focus Distribution")
    ax2.set_ylabel("")
    plt.savefig("assets/focus_pie.png")
    st.pyplot(fig2)

    # Live Emotion Trend
    st.subheader("📈 Emotion Trend Over Time")
    fig_trend, ax_trend = plt.subplots(figsize=(8, 4))
    ax_trend.plot(df["Time"], df["Emotion"], marker='o')
    ax_trend.set_xlabel("Time (s)")
    ax_trend.set_ylabel("Emotion")
    ax_trend.set_title("Emotion Changes Over Session")
    plt.xticks(rotation=45)
    plt.savefig("assets/emotion_trend.png")
    st.pyplot(fig_trend)

    # Save session to user history
    user_df = pd.read_csv(users_file)
    new_row = pd.DataFrame([{
        "Name": st.session_state.user_name,
        "Last Session": start_time.strftime("%Y-%m-%d %H:%M"),
        "Duration": duration
    }])
    user_df = pd.concat([user_df, new_row], ignore_index=True)
    user_df.to_csv(users_file, index=False)

    st.success("✅ Session saved for user!")

