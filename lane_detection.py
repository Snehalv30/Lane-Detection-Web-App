import cv2
import numpy as np

def process_frame(frame):
    """Process a single video frame and return frame with lane lines drawn"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Region of Interest (ROI)
    height = frame.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygons, 255)
    roi = cv2.bitwise_and(edges, mask)

    # Hough Transform for line detection
    lines = cv2.HoughLinesP(
        roi,
        rho=2,
        theta=np.pi/180,
        threshold=100,
        minLineLength=40,
        maxLineGap=5
    )

    # Draw lines
    line_img = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 10)

    # Overlay lines on original frame
    combined = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return combined
