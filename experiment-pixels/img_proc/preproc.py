import cv2
import numpy as np
import scipy, scipy.misc
import os
from tqdm import tqdm
from PIL import Image

def preproc(X, side):
    resized = np.array(Image.fromarray(X).resize((int(side), int(side))))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return gray/255.  # Normalize

def preproc_main(timesteps, trials):
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(this_dir, 'mov', 'extracted.mov')

    cap = cv2.VideoCapture(video_path)
    realframes = []
    frame_count = 0
    total_frames = trials * timesteps

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = cv2.VideoCapture(video_path)
                continue
        
            rgb = preproc(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 28)
            realframes.append(rgb)
            frame_count += 1
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    return realframes