import cv2
import numpy as np
import scipy, scipy.misc
import time
from tqdm import tqdm
from PIL import Image
import os, sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def preproc(X, side):
    # Resize the image
    resized = np.array(Image.fromarray(X).resize((int(side), int(side))))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return gray / 255.  # Normalize

def main(timesteps, trials):
    # Display full arrays without truncation
    np.set_printoptions(threshold=np.inf)

    video_path = '/Users/ranmaru/Documents/hnn-d/hamiltonian-nn-master/experiment-pixels/monokuro.mov'

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    realframes = []
    frame_count = 0
    total_frames = trials * timesteps

    # Prepare VideoWriter to save processed frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mov files
    output_path = 'decode.mov'
    fps = 20  # Set the desired frame rate
    width, height = 28, 28  # Set width and height for output video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if no more frames

            # Process frame
            rgb = preproc(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 28)
            realframes.append(rgb)

            # Write the processed frame to the output video
            # Convert the frame back to 8-bit format (0-255)
            out_frame = (rgb * 255).astype(np.uint8)  # Rescale to 0-255
            out.write(out_frame)

            frame_count += 1
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return realframes

if __name__ == '__main__':
    test = main(100, 200)
