import streamlit as st
import cv2
import tempfile
from lane_detection import process_frame  # our lane detection function

st.title("🚗 Lane Detection Web App")
st.write("Upload a driving video and see detected lanes in real-time!")

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    # Output video file
    out_file = "output_lane_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    st.write("⚡ Processing video... Please wait.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)
        out.write(processed)

    cap.release()
    out.release()

    st.success("✅ Processing complete!")
    st.video(out_file)
