import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

# Folders
os.makedirs("assets", exist_ok=True)
os.makedirs("reports", exist_ok=True)

st.set_page_config(page_title="AI Focus & Emotion Detector", layout="wide")
st.title("AI Focus & Emotion Detection")

if "run" not in st.session_state:
    st.session_state.run = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []
if "attention_log" not in st.session_state:
    st.session_state.attention_log = []

FRAME_WINDOW = st.image([])

# Start button
if st.button("▶️ Start Session"):
    st.session_state.run = True
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)

# Stop button
if st.button("⏹️ Stop Session"):
    st.session_state.run = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

# Main loop if running
if st.session_state.run and st.session_state.cap:
    ret, frame = st.session_state.cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        results = face_mesh.process(rgb_frame)

        # Emotion detection
        try:
            analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotions = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
            dominant_emotion = max(emotions, key=emotions.get)
        except:
            dominant_emotion = "unknown"

        st.session_state.emotion_log.append(dominant_emotion)

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

        st.session_state.attention_log.append(focus)

        # Show video feed
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Focus: {focus}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
