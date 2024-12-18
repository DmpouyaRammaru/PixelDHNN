import cv2
import numpy as np
import time
from tqdm import tqdm

def Recording(timesteps=200, trials=4):
    
    # 接続されているカメラが1つの場合，カメラのデバイスIDは0または1
    cap = cv2.VideoCapture(1)
    fps = 20
    cap.set(cv2.CAP_PROP_FPS, fps)  # カメラFPSを設定
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    frame_duration = 1.0 / fps  # 1フレームあたりの時間（秒）
    
    # カメラの幅と高さを取得
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 出力する動画の幅と高さを設定（500x500ピクセル）
    output_w, output_h = 500, 500
    
    # トリミングの開始位置を計算（中央部分をトリミング）
    x_start = max((w - output_w) // 2, 0)
    y_start = max((h - output_h) // 2, 0)
    
    # 動画保存時の形式を設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    name = "sample2.mov"
    
    # VideoWriterを設定
    video = cv2.VideoWriter(name, fourcc, fps, (output_w, output_h))
    
    print("Press 's' to start recording...")
    
    # 録画開始の待機
    while True:
        ret, frame = cap.read()  # 1フレーム読み込み
        
        if not ret:
            print("フレームの取得に失敗しました")
            break
        
        # フレームをトリミングして表示
        frame_cropped = frame[y_start:y_start + output_h, x_start:x_start + output_w]
        cv2.imshow("Press 's' to start recording", frame_cropped)

        key = cv2.waitKey(1)  # 1ミリ秒待機し、キー入力を取得
        if key == ord('s'):  # 's' キーが押されたら録画開始
            break
    
    print("Start recording")
    roop = int(trials * timesteps + fps * 10)
    
    start_time = time.time()  # 録画開始時刻を記録
    
    for i in tqdm(range(roop)):
        ret, frame = cap.read()  # 1フレーム読み込み
        
        if not ret:
            print("フレームの取得に失敗しました")
            break
        
        # フレームをトリミングして保存
        frame_cropped = frame[y_start:y_start + output_h, x_start:x_start + output_w]
        video.write(frame_cropped)
        
        # 現在の時間を取得
        elapsed_time = time.time() - start_time
        
        # 予定よりも早く進んでいる場合、フレーム間の時間を待機
        expected_time = (i + 1) * frame_duration
        if elapsed_time < expected_time:
            print("fast")
            time.sleep(expected_time - elapsed_time)  # 余った時間分スリープ

    end_time = time.time()  # 録画開始時刻を記録

    # 後処理
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Recording finished. Duration: {elapsed_time:.2f} seconds.")
    print("Capture time: {0}".format((end_time - start_time) * 1000 / roop) + "[msec]")
    return video

Recording()  # 録画を実行
