import streamlit as st
import cv2
import tempfile
from lane_detection import process_frame  # your lane detection function

st.title("🚗 Lane Detection Web App")
st.write("Upload a driving video and see lane detection live as it processes!")

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()  # placeholder for displaying video frames

    st.write("⚡ Processing and showing video live...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame (apply lane detection)
        processed = process_frame(frame)

        # Show frame in the Streamlit app
        stframe.image(processed, channels="BGR", use_column_width=True)

    cap.release()
    st.success("✅ Video finished playing!")
