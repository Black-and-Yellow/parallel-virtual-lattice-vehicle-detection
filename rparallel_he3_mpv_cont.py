import cv2
import numpy as np
import psutil
import time
import os
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

# Histogram equalization (in-place for speed)
def equalize_roi_inplace(frame, roi_x, roi_y, roi_w, roi_h):
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

# Optimized grid processing - pure NumPy vectorization
def process_grid_optimized(roi_data, grid_w, grid_h, num_rows, num_cols):
    """
    Ultra-fast grid processing using vectorized NumPy operations
    No contours, no multiprocessing overhead
    """
    result = np.zeros((num_rows, num_cols), dtype=np.int8)
    
    # Process all channels with vectorized operations
    detection_masks = []
    for channel in roi_data:
        # Vectorized blur and threshold
        blur = cv2.GaussianBlur(channel, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        detection_masks.append(dilated)
    
    # Combine masks efficiently
    if len(detection_masks) > 1:
        combined = np.all(np.stack(detection_masks), axis=0).astype(np.uint8) * 255
    else:
        combined = detection_masks[0]
    
    # Vectorized grid evaluation using array slicing
    for row in range(num_rows):
        row_start = row * grid_h
        row_end = (row + 1) * grid_h
        
        for col in range(num_cols):
            col_start = col * grid_w
            col_end = (col + 1) * grid_w
            
            # Extract cell and count white pixels (vectorized)
            cell = combined[row_start:row_end, col_start:col_end]
            white_count = np.count_nonzero(cell)
            
            # Simple threshold - faster than contours
            if white_count >= 100:
                result[row, col] = 1
    
    return result

# Generator for streaming frame pairs
def frame_pair_generator(video_source):
    """Stream frame pairs without loading all into memory"""
    is_image_sequence = os.path.isdir(video_source)
    
    if is_image_sequence:
        image_files = load_image_sequence(video_source)
        if len(image_files) < 2:
            raise ValueError("Need at least 2 images")
        
        for i in range(len(image_files) - 1):
            f1 = cv2.imread(image_files[i])
            f2 = cv2.imread(image_files[i + 1])
            if f1 is not None and f2 is not None:
                yield i + 1, f1, f2
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_source}")
        
        ret, f1 = cap.read()
        frame_idx = 1
        
        while ret:
            ret, f2 = cap.read()
            if ret:
                yield frame_idx, f1, f2
                f1 = f2
                frame_idx += 1
        
        cap.release()

def main():
    start_time = time.time()
    
    # Get video dimensions
    is_image_sequence = os.path.isdir(video_path)
    if is_image_sequence:
        image_files = load_image_sequence(video_path)
        first_frame = cv2.imread(image_files[0])
        h, w = first_frame.shape[:2]
        total_frames = len(image_files) - 1
        del first_frame
    else:
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))
    
    density_values_lane1 = []
    density_values_lane2 = []
    frame_count = 0
    
    # Process frames sequentially with optimized vectorization
    for frame_idx, f1, f2 in tqdm(frame_pair_generator(video_path), 
                                   total=total_frames, 
                                   desc="Processing frames"):
        
        if f1.shape[:2] != f2.shape[:2]:
            continue
        
        # In-place histogram equalization (no copies)
        equalize_roi_inplace(f1, roi1_x, roi1_y, roi1_width, roi1_height)
        equalize_roi_inplace(f1, roi2_x, roi2_y, roi2_width, roi2_height)
        equalize_roi_inplace(f2, roi1_x, roi1_y, roi1_width, roi1_height)
        equalize_roi_inplace(f2, roi2_x, roi2_y, roi2_width, roi2_height)
        
        # Compute difference and extract channels
        diff = cv2.absdiff(f1, f2)
        
        if channels == 'gray':
            channels_data_full = [cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)]
        else:
            hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
            channels_data_full = [cv2.split(hsv)[i] for i in channels]
        
        # Extract ROI data
        roi1_data = [ch[roi1_y:roi1_y+roi1_height, roi1_x:roi1_x+roi1_width] 
                     for ch in channels_data_full]
        roi2_data = [ch[roi2_y:roi2_y+roi2_height, roi2_x:roi2_x+roi2_width] 
                     for ch in channels_data_full]
        
        # Optimized vectorized processing (no multiprocessing overhead)
        result_matrix1 = process_grid_optimized(roi1_data, grid_width1, grid_height1, 
                                                num_rows, num_cols)
        result_matrix2 = process_grid_optimized(roi2_data, grid_width2, grid_height2, 
                                                num_rows, num_cols)
        
        # Draw rectangles in single loop
        for row in range(num_rows):
            for col in range(num_cols):
                # ROI 1
                color1 = (0, 255, 0) if result_matrix1[row, col] == 1 else (0, 0, 255)
                x1 = roi1_x + col * grid_width1
                y1 = roi1_y + row * grid_height1
                cv2.rectangle(f1, (x1, y1), 
                            (x1 + grid_width1, y1 + grid_height1), 
                            color1, 2)
                
                # ROI 2
                color2 = (0, 255, 0) if result_matrix2[row, col] == 1 else (0, 0, 255)
                x2 = roi2_x + col * grid_width2
                y2 = roi2_y + row * grid_height2
                cv2.rectangle(f1, (x2, y2), 
                            (x2 + grid_width2, y2 + grid_height2), 
                            color2, 2)
        
        # Calculate density
        density1 = np.sum(result_matrix1 == 1) / (num_rows * num_cols)
        density2 = np.sum(result_matrix2 == 1) / (num_rows * num_cols)
        density_values_lane1.append(density1)
        density_values_lane2.append(density2)
        
        # Add frame counter and write
        cv2.putText(f1, f"Frame: {frame_idx}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(f1)
        frame_count += 1
    
    out.release()
    cv2.destroyAllWindows()
    
    # Stats
    end_time = time.time()
    print(f"\n{'='*50}")
    print(f"Execution Time: {end_time - start_time:.2f} s")
    print(f"Memory Usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"Frames Processed: {frame_count}")
    print(f"Avg Density Lane1: {np.mean(density_values_lane1):.4f}")
    print(f"Avg Density Lane2: {np.mean(density_values_lane2):.4f}")
    print(f"Processing Speed: {frame_count / (end_time - start_time):.2f} FPS")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()