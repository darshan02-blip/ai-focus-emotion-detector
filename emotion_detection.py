import streamlit as st
import cv2
from deepface import DeepFace

st.set_page_config(page_title="Stage-1: Emotion Detection", layout="wide")
st.title("😊 Stage-1: Basic Emotion Detection")

FRAME_WINDOW = st.image([])
start = st.button("Start Emotion Detection")

if start:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotions = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
            dominant_emotion = max(emotions, key=emotions.get)
        except:
            dominant_emotion = "unknown"

        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
