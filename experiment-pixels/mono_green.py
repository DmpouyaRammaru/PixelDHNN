import cv2
import numpy as np
from tqdm import tqdm

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

def main():
    # 動画ファイルを読み込む
    cap = cv2.VideoCapture('sample_fixed.mov')  # 動画ファイル名を指定

    # 動画の保存設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('monokuro.mov', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    with tqdm(total=200*100) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # フレームを処理
            processed_frame = process_frame(frame)

            # 結果を表示
            cv2.imshow('Processed Frame', processed_frame)

            # 動画を書き出す
            out.write(processed_frame)
            pbar.update()

            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pbar.close()

    # リソースを解放
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
