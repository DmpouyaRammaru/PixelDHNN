import numpy as np
import cv2
import time
import csv
from tqdm import tqdm
import math
import os


this_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(this_dir, 'mov', 'for_angle.mov')
output_path = os.path.join(this_dir, 'csv', 'positions.csv')

fps = 20
timesteps = 100
frame_duration = 1.0 / fps  # 1フレームあたりの時間（秒）
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FPS, fps)  # カメラFPSを設定

frame_count = 0
positions = []

# 角度の最大値と最小値を初期化
max_angle = -float('inf')
min_angle = float('inf')

for i in tqdm(range(timesteps)):
    ret, frame = cap.read()
    if not ret:
        print("ビデオ終了")
        break
    
    frame_count += 1

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 色範囲の設定
    red_lower1 = np.array([0, 150, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 150, 50])
    red_upper2 = np.array([180, 255, 255])
    blue_lower = np.array([100, 150, 50])
    blue_upper = np.array([140, 255, 255])

    # 色のマスク
    red_img1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_img2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_img = red_img1 + red_img2
    blue_img = cv2.inRange(hsv, blue_lower, blue_upper)

    # 座標を抽出
    blue_x = np.argmax(np.sum(blue_img, axis=0))
    blue_y = np.argmax(np.sum(blue_img, axis=1))
    red_x = np.argmax(np.sum(red_img, axis=0))
    red_y = np.argmax(np.sum(red_img, axis=1))

    # 座標差分 (青点 - 赤点)
    dx = blue_x - red_x
    dy = blue_y - red_y

    # 赤点を基準に垂直方向の軸との角度を計算（ラジアン -> 度に変換）
    angle = math.atan2(dx, dy)  # dx, dyを入れ替えることで基準を垂直方向に設定
    angle_deg = math.degrees(angle)  # 度

    # 最大・最小角度の更新
    max_angle = max(max_angle, angle_deg)
    min_angle = min(min_angle, angle_deg)

    # 保存
    positions.append((frame_count, dx, dy, angle_deg))

    # 抽出した座標に丸を描く
    cv2.circle(frame, (blue_x, blue_y), 10, (255, 0, 0), -1)  # 青
    cv2.circle(frame, (red_x, red_y), 10, (0, 0, 255), -1)  # 赤

    # 表示
    cv2.imshow('frame', frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Delta X", "Delta Y", "Angle (degrees)"])
    writer.writerows(positions)

cap.release()
cv2.destroyAllWindows()
f.close()
