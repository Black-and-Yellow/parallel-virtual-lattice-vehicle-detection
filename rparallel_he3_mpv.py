import cv2
import numpy as np
import psutil
import time
import os
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

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

# Map color channel
choices = {
    'H': [0],
    'S': [1],
    'V': [2],
    'H+S': [0, 1],
    'H+V': [0, 2],
    'S+V': [1, 2],
    'H+S+V': [0, 1, 2],
    'gray': 'gray'
}
channels = choices[color_channel]

# Load video or image sequence
def load_image_sequence(path):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(path) if f.lower().endswith(exts)]
    files.sort()
    return [os.path.join(path, f) for f in files]

# Vectorized histogram equalization for ROI
def equalize_roi_vectorized(frame, roi_x, roi_y, roi_w, roi_h):
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    return frame

# Vectorized grid processing - processes entire grid at once
def process_grid_vectorized(roi_data, grid_w, grid_h, num_rows, num_cols):
    """
    Vectorized grid processing using NumPy operations
    roi_data: list of channel data for the ROI
    """
    result = np.zeros((num_rows, num_cols), dtype=int)
    
    for channel in roi_data:
        # Apply Gaussian blur to entire channel at once
        blur = cv2.GaussianBlur(channel, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        
        # Vectorized processing for all grid cells
        for row in range(num_rows):
            for col in range(num_cols):
                y_start = row * grid_h
                y_end = (row + 1) * grid_h
                x_start = col * grid_w
                x_end = (col + 1) * grid_w
                
                cell = dilated[y_start:y_end, x_start:x_end]
                
                # Vectorized contour area check
                contours, _ = cv2.findContours(cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if any(cv2.contourArea(c) >= 100 for c in contours):
                    result[row, col] = 1
    
    return result

# Process a single frame pair (for multiprocessing)
def process_frame_pair(args):
    """
    Process a single frame pair
    Returns: (frame_idx, annotated_frame, density1, density2, result_matrix1, result_matrix2)
    """
    frame_idx, f1, f2 = args
    
    if f1.shape[:2] != f2.shape[:2]:
        return None
    
    # Histogram equalization on both ROIs
    f1_eq = f1.copy()
    f2_eq = f2.copy()
    
    equalize_roi_vectorized(f1_eq, roi1_x, roi1_y, roi1_width, roi1_height)
    equalize_roi_vectorized(f1_eq, roi2_x, roi2_y, roi2_width, roi2_height)
    equalize_roi_vectorized(f2_eq, roi1_x, roi1_y, roi1_width, roi1_height)
    equalize_roi_vectorized(f2_eq, roi2_x, roi2_y, roi2_width, roi2_height)
    
    # Channel selection and difference
    diff = cv2.absdiff(f1_eq, f2_eq)
    
    if channels == 'gray':
        channels_data_full = [cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)]
    else:
        hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
        channels_data_full = [cv2.split(hsv)[i] for i in channels]
    
    # Extract ROI data for vectorized processing
    roi1_data = [ch[roi1_y:roi1_y+roi1_height, roi1_x:roi1_x+roi1_width] for ch in channels_data_full]
    roi2_data = [ch[roi2_y:roi2_y+roi2_height, roi2_x:roi2_x+roi2_width] for ch in channels_data_full]
    
    # Vectorized grid processing
    result_matrix1 = process_grid_vectorized(roi1_data, grid_width1, grid_height1, num_rows, num_cols)
    result_matrix2 = process_grid_vectorized(roi2_data, grid_width2, grid_height2, num_rows, num_cols)
    
    # Draw rectangles on original frame
    for row in range(num_rows):
        for col in range(num_cols):
            color1 = (0, 255, 0) if result_matrix1[row, col] == 1 else (0, 0, 255)
            color2 = (0, 255, 0) if result_matrix2[row, col] == 1 else (0, 0, 255)
            
            cv2.rectangle(f1, 
                         (roi1_x + col * grid_width1, roi1_y + row * grid_height1),
                         (roi1_x + (col + 1) * grid_width1, roi1_y + (row + 1) * grid_height1), 
                         color1, 2)
            cv2.rectangle(f1, 
                         (roi2_x + col * grid_width2, roi2_y + row * grid_height2),
                         (roi2_x + (col + 1) * grid_width2, roi2_y + (row + 1) * grid_height2), 
                         color2, 2)
    
    # Calculate density
    density1 = np.sum(result_matrix1 == 1) / (num_rows * num_cols)
    density2 = np.sum(result_matrix2 == 1) / (num_rows * num_cols)
    
    cv2.putText(f1, f"Frame: {frame_idx}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return (frame_idx, f1, density1, density2, result_matrix1, result_matrix2)

def main():
    start_time = time.time()
    
    # Load all frame pairs first
    is_image_sequence = os.path.isdir(video_path)
    frame_pairs = []
    
    if is_image_sequence:
        image_files = load_image_sequence(video_path)
        if len(image_files) < 2:
            raise ValueError("Need at least 2 images in the sequence")
        
        print("Loading image sequence...")
        for i in range(len(image_files) - 1):
            f1 = cv2.imread(image_files[i])
            f2 = cv2.imread(image_files[i + 1])
            if f1 is not None and f2 is not None:
                frame_pairs.append((i + 1, f1, f2))
        
        h, w = frame_pairs[0][1].shape[:2]
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("Loading video frames...")
        ret, f1 = cap.read()
        frame_idx = 1
        while ret:
            ret, f2 = cap.read()
            if ret:
                frame_pairs.append((frame_idx, f1.copy(), f2.copy()))
                f1 = f2
                frame_idx += 1
        
        cap.release()
    
    print(f"Loaded {len(frame_pairs)} frame pairs")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))
    
    # Process frames in parallel using multiprocessing
    num_processes = max(1, cpu_count() - 1)  # Leave one core free
    print(f"Processing with {num_processes} processes...")
    
    density_values_lane1 = []
    density_values_lane2 = []
    
    # Use multiprocessing Pool
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_frame_pair, frame_pairs),
            total=len(frame_pairs),
            desc="Processing frames"
        ))
    
    # Filter out None results and sort by frame index
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[0])
    
    # Write frames and collect densities
    print("Writing output video...")
    for frame_idx, frame, density1, density2, _, _ in tqdm(results, desc="Writing frames"):
        density_values_lane1.append(density1)
        density_values_lane2.append(density2)
        out.write(frame)
    
    out.release()
    cv2.destroyAllWindows()
    
    # Stats
    end_time = time.time()
    print(f"\n{'='*50}")
    print(f"Execution Time: {end_time - start_time:.2f} s")
    print(f"Memory Usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"Frames Processed: {len(results)}")
    print(f"Avg Density Lane1: {np.mean(density_values_lane1):.4f}")
    print(f"Avg Density Lane2: {np.mean(density_values_lane2):.4f}")
    print(f"Processing Speed: {len(results) / (end_time - start_time):.2f} FPS")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()