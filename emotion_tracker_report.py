import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit page setup
st.set_page_config(page_title="Emotion Tracker", layout="wide")
st.title("🧠 Real-Time Emotion Tracker")

# Define emotion color mapping
color_map = {
    'happy': 'yellow',
    'sad': 'blue',
    'angry': 'red',
    'neutral': 'gray',
    'surprised': 'orange',
    'fear': 'purple',
    'disgust': 'green'
}

# Initialize session state DataFrame
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['timestamp', 'emotion'])

# Simulate Emotion Detection (you can replace this with actual model logic)
st.subheader("Simulate Emotion Detection")
simulated_emotion = st.selectbox("Choose emotion", list(color_map.keys()))
if st.button("Log Emotion"):
    current_time = pd.Timestamp.now()
    new_row = pd.DataFrame([{
        'timestamp': current_time,
        'emotion': simulated_emotion
    }])
    st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
    st.success(f"✅ Logged emotion: {simulated_emotion}")

# Get the current DataFrame
df = st.session_state.df

# Map colors safely
if 'emotion' in df.columns:
    df['color'] = df['emotion'].map(color_map).fillna('black')
else:
    st.warning("⚠️ 'emotion' column not found.")
    df['color'] = 'black'

# Show emotion log table
st.subheader("📄 Logged Emotions")
st.dataframe(df)

# Emotion Frequency Plot
if not df.empty and 'emotion' in df.columns:
    st.subheader("📊 Emotion Frequency")
    fig1, ax1 = plt.subplots()
    emotion_counts = df['emotion'].value_counts()
    bar_colors = [color_map.get(emotion, 'black') for emotion in emotion_counts.index]
    ax1.bar(emotion_counts.index, emotion_counts.values, color=bar_colors)
    ax1.set_xlabel("Emotion")
    ax1.set_ylabel("Count")
    ax1.set_title("Emotion Frequency")
    st.pyplot(fig1)

# Emotion Over Time Plot
if not df.empty and {'timestamp', 'emotion'}.issubset(df.columns):
    st.subheader("🕒 Emotions Over Time")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['timestamp'], df['emotion'], c=df['color'])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Emotion")
    ax2.set_title("Emotion Timeline")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
