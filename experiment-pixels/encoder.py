import cv2
import numpy as np
from tqdm import tqdm
import csv

def detect_and_track(video_path, output_csv):
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    positions = []

    red_lower1 = np.array([0, 150, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 150, 50])
    red_upper2 = np.array([180, 255, 255])

    green_lower = np.array([40, 150, 50])
    green_upper = np.array([80, 255, 255])

    frame_count = 0
    # for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        _, red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        origin_x, origin_y = None, None
        for contour in red_contours:
            if cv2.contourArea(contour) > 500:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    origin_x = int(M["m10"] / M["m00"])
                    origin_y = int(M["m01"] / M["m00"])
                    break

        if origin_x is None or origin_y is None:
            print(f"Frame {frame_count}: Red point not found.")
            continue

        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        result = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(result) == 2:
            green_contours, _ = result
        else:
            _, green_contours, _ = result

        green_x, green_y = None, None
        for contour in green_contours:
            if cv2.contourArea(contour) > 500:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    green_x = int(M["m10"] / M["m00"])
                    green_y = int(M["m01"] / M["m00"])
                    break

        if green_x is None or green_y is None:
            print(f"Frame {frame_count}: Green point not found.")
            continue

        dx = green_x - origin_x
        dy = green_y - origin_y
        positions.append((frame_count, dx, dy))

        frame_with_points = frame.copy()
        cv2.circle(frame_with_points, (origin_x, origin_y), 5, (0, 0, 255), -1)
        cv2.circle(frame_with_points, (green_x, green_y), 5, (0, 255, 0), -1)
        cv2.line(frame_with_points, (origin_x, origin_y), (green_x, green_y), (255, 0, 0), 2)
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Delta X", "Delta Y"])
        writer.writerows(positions)
    print(f"Results saved to {output_csv}")

video_path = "your_video_path.mov"
output_csv = "tracked_positions.csv"

detect_and_track(video_path, output_csv)
