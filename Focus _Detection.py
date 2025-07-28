import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp

st.set_page_config(page_title="Stage-2: Emotion + Focus", layout="wide")
st.title("🧠 Stage-2: Emotion + Focus Detection")

FRAME_WINDOW = st.image([])
start = st.button("Start Detection")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def ear(eye):
    A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
    B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
    C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
    return (A + B) / (2.0 * C)

if start:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        try:
            analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotions = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
            dominant_emotion = max(emotions, key=emotions.get)
        except:
            dominant_emotion = "unknown"

        focus = "Not Focused"
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            left_eye = [landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [landmarks.landmark[i] for i in [263, 387, 385, 362, 380, 373]]
            avg_ear = (ear(left_eye) + ear(right_eye)) / 2.0
            if avg_ear > 0.18:
                focus = "Focused"

        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Focus: {focus}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
