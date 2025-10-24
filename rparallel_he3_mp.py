import cv2
import numpy as np
import psutil
import time
import os
import pandas as pd
from openpyxl import load_workbook, Workbook
from concurrent.futures import ThreadPoolExecutor
import json
from tqdm import tqdm

# Load configuration
with open("user_input_data.json", "r") as file:
    config = json.load(file)

video_path = config["video"]
color_channel = config["color_channel"]
rows, cols = config["grids"]["rows"], config["grids"]["cols"]

# ROIs
roi1_x, roi1_y, roi1_width, roi1_height = 545, 159, 284, 140
roi2_x, roi2_y, roi2_width, roi2_height = 238, 161, 284, 140

num_rows, num_cols = rows, cols
grid_width1 = roi1_width // num_cols
grid_height1 = roi1_height // num_rows
grid_width2 = roi2_width // num_cols
grid_height2 = roi2_height // num_rows

density_values_lane1 = []
density_values_lane2 = []

frame_count = 0
start_time = time.time()

result_matrix1 = np.zeros((num_rows, num_cols), dtype=int)
result_matrix2 = np.zeros((num_rows, num_cols), dtype=int)

# Load video or image sequence
def load_image_sequence(path):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(path) if f.lower().endswith(exts)]
    files.sort()
    return [os.path.join(path, f) for f in files]

is_image_sequence = os.path.isdir(video_path)
image_files = []
cap = None
if is_image_sequence:
    image_files = load_image_sequence(video_path)
    if len(image_files) < 2:
        raise ValueError("Need at least 2 images in the sequence")
    first_frame = cv2.imread(image_files[0])
    h, w = first_frame.shape[:2]
else:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))

# Generator for consecutive frames
def frame_pair_generator():
    if is_image_sequence:
        for i in range(len(image_files)-1):
            f1 = cv2.imread(image_files[i])
            f2 = cv2.imread(image_files[i+1])
            if f1 is None or f2 is None:
                continue
            yield f1, f2
    else:
        ret, f1 = cap.read()
        if not ret:
            return
        ret, f2 = cap.read()
        while ret and f1 is not None and f2 is not None:
            yield f1, f2
            f1 = f2
            ret, f2 = cap.read()

# Histogram equalization for ROI
def equalize_roi(frame, roi_x, roi_y, roi_w, roi_h):
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    return frame

def apply_hist_parallel(frame):
    with ThreadPoolExecutor() as executor:
        executor.submit(equalize_roi, frame, roi1_x, roi1_y, roi1_width, roi1_height)
        executor.submit(equalize_roi, frame, roi2_x, roi2_y, roi2_width, roi2_height)
    return frame

# Process channel
def process_channel(channel):
    blur = cv2.GaussianBlur(channel, (5,5),0)
    _, thresh = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Process grid cell
def process_grid_cell(grid_x, grid_y, grid_w, grid_h, channels_data):
    detection_flags = []
    for channel in channels_data:
        cell = channel[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w]
        contours = process_channel(cell)
        detection_flags.append(any(cv2.contourArea(c)>=100 for c in contours))
    return int(all(detection_flags))

def process_grid(roi_x, roi_y, grid_w, grid_h, channels_data):
    result = np.zeros((num_rows,num_cols),dtype=int)
    args = [(roi_x+col*grid_w, roi_y+row*grid_h, grid_w, grid_h, channels_data)
            for row in range(num_rows) for col in range(num_cols)]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda p: process_grid_cell(*p), args))
    for idx, val in enumerate(results):
        row = idx//num_cols
        col = idx%num_cols
        result[row,col] = val
    return result

# Map color channel
choices = {
    'H':[0],
    'S':[1],
    'V':[2],
    'H+S':[0,1],
    'H+V':[0,2],
    'S+V':[1,2],
    'H+S+V':[0,1,2],
    'gray':'gray'
}

channels = choices[color_channel]

# Main processing
def main():
    global frame_count
    for f1, f2 in tqdm(frame_pair_generator(), desc="Processing frames"):
        frame_count += 1

        if f1.shape[:2] != f2.shape[:2]:
            continue

        # Histogram equalization
        f1_eq = apply_hist_parallel(f1.copy())
        f2_eq = apply_hist_parallel(f2.copy())

        # Channel selection
        if channels=='gray':
            diff = cv2.absdiff(f1_eq, f2_eq)
            channels_data = [cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)]
        else:
            diff = cv2.absdiff(f1_eq, f2_eq)
            hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
            channels_data = [cv2.split(hsv)[i] for i in channels]

        # Process both grids
        result_matrix1 = process_grid(roi1_x, roi1_y, grid_width1, grid_height1, channels_data)
        result_matrix2 = process_grid(roi2_x, roi2_y, grid_width2, grid_height2, channels_data)

        # Draw rectangles
        for row in range(num_rows):
            for col in range(num_cols):
                color1 = (0,255,0) if result_matrix1[row,col]==1 else (0,0,255)
                color2 = (0,255,0) if result_matrix2[row,col]==1 else (0,0,255)
                cv2.rectangle(f1, (roi1_x+col*grid_width1, roi1_y+row*grid_height1),
                             (roi1_x+(col+1)*grid_width1, roi1_y+(row+1)*grid_height1), color1, 2)
                cv2.rectangle(f1, (roi2_x+col*grid_width2, roi2_y+row*grid_height2),
                             (roi2_x+(col+1)*grid_width2, roi2_y+(row+1)*grid_height2), color2, 2)

        # Density
        density_values_lane1.append(np.sum(result_matrix1==1)/(num_rows*num_cols))
        density_values_lane2.append(np.sum(result_matrix2==1)/(num_rows*num_cols))

        cv2.putText(f1, f"Frame: {frame_count}", (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        out.write(f1)

    # Stats
    end_time = time.time()
    print(f"Execution Time: {end_time-start_time:.2f} s")
    print(f"Memory Usage: {psutil.Process().memory_info().rss/(1024*1024):.2f} MB")
    print(f"Avg Density Lane1: {np.mean(density_values_lane1):.4f}")
    print(f"Avg Density Lane2: {np.mean(density_values_lane2):.4f}")
    if cap is not None: cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Frames Processed: {frame_count}")

if __name__=='__main__':
    main()
