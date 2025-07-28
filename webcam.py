import streamlit as st
import cv2

st.set_page_config(page_title="Stage-0: Webcam Capture", layout="wide")
st.title("📷 Stage-0: Basic Webcam Capture")

FRAME_WINDOW = st.image([])
start = st.button("Start Camera")

if start:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
