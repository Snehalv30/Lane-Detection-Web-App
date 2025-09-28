import cv2
import numpy as np

def process_frame(frame):
    height, width = frame.shape[:2]

    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)

    # 4. Mask Region of Interest (ROI)
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width*0.55), int(height*0.6)),
        (int(width*0.45), int(height*0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)

    # 5. Hough Transform to detect lines
    lines = cv2.HoughLinesP(
        masked,
        rho=2,
        theta=np.pi/180,
        threshold=100,
        minLineLength=40,
        maxLineGap=50
    )

    # 6. Separate left and right lanes
    left, right = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # avoid division by zero
            if abs(slope) < 0.5:  # ignore nearly horizontal lines
                continue
            if slope < 0:  # left lane
                left.append(line[0])
            else:          # right lane
                right.append(line[0])

    # 7. Draw averaged lanes
    lane_image = np.zeros_like(frame)
    def draw_lines(img, lines, color):
        if len(lines) > 0:
            x_coords, y_coords = [], []
            for x1, y1, x2, y2 in lines:
                x_coords += [x1, x2]
                y_coords += [y1, y2]
            poly = np.polyfit(y_coords, x_coords, 1)  # fit line
            y1, y2 = height, int(height*0.6)
            x1 = int(np.polyval(poly, y1))
            x2 = int(np.polyval(poly, y2))
            cv2.line(img, (x1, y1), (x2, y2), color, 10)

    draw_lines(lane_image, left, (0, 255, 0))
    draw_lines(lane_image, right, (0, 255, 0))

    # 8. Overlay result on original frame
    combined = cv2.addWeighted(frame, 0.8, lane_image, 1, 1)

    return combined
