import streamlit as st
import cv2
import tempfile
import os
from lane_detection import process_frame  # your lane detection function

st.title("🚗 Lane Detection Web App")
st.write("Upload a driving video and see detected lanes in real-time!")

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)

    # Check video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    out_file = "output_lane_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # better codec for browser
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    st.write("⚡ Processing video... Please wait...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)
        out.write(processed)

    cap.release()
    out.release()

    st.success("✅ Processing complete!")

    # Ensure file is written before reading
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        with open(out_file, "rb") as f:
            st.video(f.read())
    else:
        st.error("⚠️ Processed video could not be generated. Check FPS/codec.")
