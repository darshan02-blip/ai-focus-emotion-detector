import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import time

st.set_page_config(page_title="AI Focus & Emotion Detector", layout="wide")
st.title("🧠 AI Focus & Emotion Detection")

start = st.button("Start Detection")
FRAME_WINDOW = st.image([])

if start:
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    start_time = time.time()
    duration = 10  # seconds
    emotion_counts = {}
    focused_frames = 0
    total_frames = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to access camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        focus_status = "No Face"
        emotion_status = "..."

        if results.multi_face_landmarks:
            focus_status = "Face Detected"
            focused_frames += 1
            try:
                result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                dominant = result[0]["dominant_emotion"]
                emotion_status = dominant.capitalize()

                if emotion_status in emotion_counts:
                    emotion_counts[emotion_status] += 1
                else:
                    emotion_counts[emotion_status] = 1

            except:
                emotion_status = "No Emotion"

        total_frames += 1
        cv2.putText(frame, f"Focus: {focus_status}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        cv2.putText(frame, f"Emotion: {emotion_status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()

    # Summary Block
    st.warning("📸 Detection ended after 10 seconds.")
    st.subheader("📊 Session Summary")

    if total_frames > 0:
        focus_score = (focused_frames / total_frames) * 100
        st.success(f"🎯 Focus Score: {focus_score:.2f}%")
    else:
        st.error("⚠️ No frames captured.")

    if emotion_counts:
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        st.success(f"😊 Dominant Emotion: {dominant_emotion}")
        st.markdown("**📋 Emotion Breakdown:**")
        for emotion, count in emotion_counts.items():
            st.markdown(f"- {emotion}: {count} frames")
    else:
        st.info("No emotions were detected.") 

