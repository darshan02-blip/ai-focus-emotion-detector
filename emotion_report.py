import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import time

# Create folders if not exist
os.makedirs("assets", exist_ok=True)
os.makedirs("reports", exist_ok=True)

st.set_page_config(page_title="AI Focus & Emotion Detector", layout="wide")
st.title("🧠 AI Focus & Emotion Detection")

# Initialize session state
if "run" not in st.session_state:
    st.session_state.run = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []
if "attention_log" not in st.session_state:
    st.session_state.attention_log = []

frame_window = st.empty()
fps = st.slider("Frame Rate (FPS)", 1, 15, 5)
resolution = st.selectbox("Camera Resolution", ["480p", "720p"])

# PDF Report Generator
def generate_pdf_report(summary_file, emotion_img, focus_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="AI Focus & Emotion Session Report", ln=True, align="C")

    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            for line in f:
                safe_line = line.encode("latin-1", errors="ignore").decode("latin-1")
                pdf.multi_cell(0, 10, txt=safe_line)

    if os.path.exists(emotion_img):
        pdf.add_page()
        pdf.image(emotion_img, x=30, y=30, w=150)

    if os.path.exists(focus_img):
        pdf.add_page()
        pdf.image(focus_img, x=30, y=30, w=150)

    output_path = "reports/session_report.pdf"
    pdf.output(output_path)
    return output_path

# Start Session
if st.button("▶️ Start Session"):
    st.session_state.run = True
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        if resolution == "480p":
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Stop Session
if st.button("⏹️ Stop Session"):
    st.session_state.run = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

# Process one frame per rerun
if st.session_state.run and st.session_state.cap:
    ret, frame = st.session_state.cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        results = face_mesh.process(rgb_frame)

        # Emotion Detection
        try:
            analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotions = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
            dominant_emotion = max(emotions, key=emotions.get)
        except:
            dominant_emotion = "unknown"

        st.session_state.emotion_log.append(dominant_emotion)

        # Focus Detection
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

        st.session_state.attention_log.append(focus)

        # Display video feed
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Focus: {focus}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    time.sleep(1 / fps)
    st.rerun()  # Auto-refresh for next frame

# Generate summary when stopped
if not st.session_state.run and st.session_state.emotion_log:
    df = pd.DataFrame({"Emotion": st.session_state.emotion_log, "Focus": st.session_state.attention_log})
    df.to_csv("emotion_log.csv", index=False)

    with open("session_summary.txt", "w", encoding="utf-8") as f:
        f.write("Emotion Summary:\n")
        for emotion, percent in df["Emotion"].value_counts(normalize=True).items():
            f.write(f"{emotion}: {percent*100:.2f}%\n")
        f.write("\nFocus Summary:\n")
        for focus, percent in df["Focus"].value_counts(normalize=True).items():
            f.write(f"{focus}: {percent*100:.2f}%\n")

    # Charts
    emotion_summary = df["Emotion"].value_counts(normalize=True) * 100
    focus_summary = df["Focus"].value_counts(normalize=True) * 100

    fig1, ax1 = plt.subplots()
    emotion_summary.plot.pie(autopct="%1.1f%%", ax=ax1)
    ax1.set_title("Emotion Distribution")
    ax1.set_ylabel("")
    plt.savefig("assets/emotion_pie.png")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    focus_summary.plot.pie(autopct="%1.1f%%", ax=ax2)
    ax2.set_title("Focus Distribution")
    ax2.set_ylabel("")
    plt.savefig("assets/focus_pie.png")
    st.pyplot(fig2)

    if st.button("📤 Generate PDF Report"):
        path = generate_pdf_report("session_summary.txt", "assets/emotion_pie.png", "assets/focus_pie.png")
        st.success(f"✅ PDF Generated: {path}")
        with open(path, "rb") as file:
            st.download_button("⬇️ Download Report", file, file_name=os.path.basename(path))
