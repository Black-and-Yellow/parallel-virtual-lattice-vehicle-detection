import cv2
import numpy as np
import psutil
import time
import os
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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

# Vectorized histogram equalization for ROI (in-place)
def equalize_roi_inplace(frame, roi_x, roi_y, roi_w, roi_h):
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

# Ultra-fast vectorized grid processing - no contours, pure NumPy
def process_grid_fast(roi_data, grid_w, grid_h, num_rows, num_cols):
    """
    Fast vectorized grid processing using NumPy thresholding
    Avoids expensive contour detection
    """
    result = np.zeros((num_rows, num_cols), dtype=np.int8)  # Use int8 to save memory
    
    detection_masks = []
    for channel in roi_data:
        # Vectorized operations on entire channel
        blur = cv2.GaussianBlur(channel, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        detection_masks.append(dilated)
    
    # Stack masks for efficient computation
    if len(detection_masks) > 1:
        combined_mask = np.all(np.stack(detection_masks), axis=0).astype(np.uint8) * 255
    else:
        combined_mask = detection_masks[0]
    
    # Vectorized grid cell evaluation
    for row in range(num_rows):
        for col in range(num_cols):
            y_start = row * grid_h
            y_end = (row + 1) * grid_h
            x_start = col * grid_w
            x_end = (col + 1) * grid_w
            
            cell = combined_mask[y_start:y_end, x_start:x_end]
            
            # Fast pixel count instead of contour detection
            if np.sum(cell > 0) >= 100:  # At least 100 white pixels
                result[row, col] = 1
    
    return result

# Process a single frame pair (for multiprocessing)
def process_frame_pair(args):
    """
    Process a single frame pair with minimal memory overhead
    Returns: (frame_idx, annotated_frame, density1, density2)
    """
    frame_idx, f1, f2 = args
    
    if f1.shape[:2] != f2.shape[:2]:
        return None
    
    # In-place histogram equalization (no copy)
    equalize_roi_inplace(f1, roi1_x, roi1_y, roi1_width, roi1_height)
    equalize_roi_inplace(f1, roi2_x, roi2_y, roi2_width, roi2_height)
    equalize_roi_inplace(f2, roi1_x, roi1_y, roi1_width, roi1_height)
    equalize_roi_inplace(f2, roi2_x, roi2_y, roi2_width, roi2_height)
    
    # Channel selection and difference
    diff = cv2.absdiff(f1, f2)
    
    if channels == 'gray':
        channels_data_full = [cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)]
    else:
        hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
        channels_data_full = [cv2.split(hsv)[i] for i in channels]
    
    # Extract ROI data for vectorized processing
    roi1_data = [ch[roi1_y:roi1_y+roi1_height, roi1_x:roi1_x+roi1_width] for ch in channels_data_full]
    roi2_data = [ch[roi2_y:roi2_y+roi2_height, roi2_x:roi2_x+roi2_width] for ch in channels_data_full]
    
    # Fast vectorized grid processing
    result_matrix1 = process_grid_fast(roi1_data, grid_width1, grid_height1, num_rows, num_cols)
    result_matrix2 = process_grid_fast(roi2_data, grid_width2, grid_height2, num_rows, num_cols)
    
    # Draw rectangles on frame (reuse f1)
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
    
    return (frame_idx, f1, density1, density2)

# Generator for streaming frame pairs in batches
def batch_frame_generator(video_source, batch_size=64):
    """
    Generator that yields batches of frame pairs
    Keeps memory usage constant
    """
    is_image_sequence = os.path.isdir(video_source)
    
    if is_image_sequence:
        image_files = load_image_sequence(video_source)
        if len(image_files) < 2:
            raise ValueError("Need at least 2 images in the sequence")
        
        batch = []
        for i in range(len(image_files) - 1):
            f1 = cv2.imread(image_files[i])
            f2 = cv2.imread(image_files[i + 1])
            if f1 is not None and f2 is not None:
                batch.append((i + 1, f1, f2))
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        
        if batch:  # Yield remaining frames
            yield batch
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_source}")
        
        ret, f1 = cap.read()
        if not ret:
            cap.release()
            return
        
        frame_idx = 1
        batch = []
        
        while True:
            ret, f2 = cap.read()
            if not ret:
                break
            
            batch.append((frame_idx, f1, f2))
            f1 = f2
            frame_idx += 1
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:  # Yield remaining frames
            yield batch
        
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
        del first_frame  # Free memory
    else:
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))
    
    # Process frames in batches with multiprocessing
    batch_size = 64  # Small batch = low memory
    num_processes = max(1, cpu_count() // 2)  # Use half cores to avoid memory overload
    print(f"Processing with {num_processes} processes, batch size: {batch_size}")
    
    density_values_lane1 = []
    density_values_lane2 = []
    frame_count = 0
    
    # Stream and process batches
    with Pool(processes=num_processes) as pool:
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        for batch in batch_frame_generator(video_path, batch_size):
            # Process batch in parallel
            results = pool.map(process_frame_pair, batch)
            
            # Filter None results and sort by frame index
            results = [r for r in results if r is not None]
            results.sort(key=lambda x: x[0])
            
            # Write frames immediately (don't store)
            for frame_idx, frame, density1, density2 in results:
                density_values_lane1.append(density1)
                density_values_lane2.append(density2)
                out.write(frame)
                frame_count += 1
            
            pbar.update(len(results))
            
            # Explicitly delete batch to free memory
            del batch
            del results
        
        pbar.close()
    
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