import cv2
import numpy as np
import streamlit as st
import tempfile

st.set_page_config(page_title="Lane Detection App", layout="wide")

st.title("🚘 Lane Detection Web App")
st.write("Upload a driving video and see road lanes detected in real-time!")

# Lane detection functions
def canny_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2, int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def detect_lines(frame, edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=150)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return frame


uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()  # placeholder for video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        edges = canny_edge_detection(frame)
        cropped = region_of_interest(edges)
        lane_frame = detect_lines(frame, cropped)

        # Convert BGR → RGB for Streamlit
        lane_frame = cv2.cvtColor(lane_frame, cv2.COLOR_BGR2RGB)

        # Display frame by frame like a video
        stframe.image(lane_frame, channels="RGB", use_container_width=True)

    cap.release()
