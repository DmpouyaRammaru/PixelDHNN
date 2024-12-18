import cv2
import numpy as np
from tqdm import tqdm

trials = 200
points = 100
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

cap = cv2.VideoCapture('sample_trimed.mov')  # 動画ファイル名を指定
out = cv2.VideoWriter('sample_fixed.mov', fourcc, fps, (int(cap.get(3)), int(cap.get(4)))) # ファイル名，保存形式，動画の高さ，幅，fpsを指定．

for _ in tqdm(range(trials)):
    for _ in range(points):
        ret,frame = cap.read()
        if not ret:
            break
        out.write(frame)
    for _ in range(200 - points):
        ret,frame = cap.read()
        if not ret:
            break

# リソースを解放
cap.release()
out.release()
cv2.destroyAllWindows()
