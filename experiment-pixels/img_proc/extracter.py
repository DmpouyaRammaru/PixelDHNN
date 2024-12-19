import cv2
import numpy as np
from tqdm import tqdm
import os
this_dir = os.path.dirname(os.path.abspath(__file__))



def trime_frame():
    video_path = os.path.join(this_dir, 'mov', 'for_extracter.mov')
    out_path = os.path.join(this_dir, 'mov', 'not_used.mov')
    trials = 200
    points = 100
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(out_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4)))) 

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

def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([28, 65, 65])
    upper_green = np.array([ 90, 255, 255])

    # mask:緑，mask_inv:緑以外
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    black_background = np.zeros_like(frame)
    white_background = np.full_like(frame, 255)

    # 黒色の背景と赤色部分を合成
    green_to_black = cv2.bitwise_and(white_background, white_background, mask=mask)
    result = cv2.add(green_to_black, cv2.bitwise_and(black_background, black_background, mask=mask_inv))

    return result

def extracter():
    video_path = os.path.join(this_dir, 'mov', 'not_used.mov')
    out_path = os.path.join(this_dir, 'mov', 'extracted.mov')
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    with tqdm(total=200*100) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # フレームを処理
            processed_frame = process_frame(frame)

            # 動画を書き出す
            out.write(processed_frame)
            pbar.update()

        pbar.close()

    # リソースを解放
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    trime_frame()
    extracter()
