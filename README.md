
# parallel-virtual-lattice-vehicle-detection

## Project Overview

This project provides a set of Python scripts for analyzing He3 video and image data using grid-based motion and change detection, rather than deep learning object detection algorithms like YOLO.

### What Are We Achieving?
- The goal is to efficiently detect and analyze changes, density, and motion in specific regions of interest (ROIs) within video or image sequences, especially for He3 experiments or traffic density analysis.
- The approach divides ROIs into grids and uses image processing techniques (histogram equalization, frame differencing, contour analysis, and pixel counting) to detect changes and compute density.
- The scripts support both sequential and parallel processing, with optimizations for speed and memory usage.

### Why Use This Instead of YOLO or Other Deep Learning Algorithms?
- **No Training Required:** Unlike YOLO or other deep learning models, this approach does not require labeled training data or model training, making it suitable for custom or scientific datasets where annotated data is limited or unavailable.
- **Explainability:** The grid-based method provides transparent, interpretable results for each cell, which is important for scientific analysis and reporting.
- **Speed and Resource Efficiency:** These scripts are optimized for fast processing and low memory usage, and can run on standard hardware without GPU acceleration.
- **Customizability:** The method allows easy adjustment of grid size, ROI selection, and detection thresholds to fit specific experimental needs.
- **Focus on Change/Motion:** The primary goal is to detect changes or motion in regions, not to classify or localize objects, which is more appropriate for He3 and density analysis tasks.

This makes the project ideal for scenarios where deep learning is impractical or unnecessary, and where fast, explainable, and customizable analysis is required.

---

This repository contains several Python scripts for parallel and sequential processing related to He3 data analysis. Below is an explanation of each script, its purpose, and how it improves upon the base scripts.

## Base Scripts

### `seq.py`
- **Description:** Implements the base sequential algorithm for He3 video/image sequence processing.
- **Processing Steps:**
	1. Loads configuration from a JSON file, including video path, color channel, and grid settings.
	2. Supports both video files and image sequences as input.
	3. For each pair of consecutive frames, applies histogram equalization to the region of interest (ROI).
	4. Computes the difference between frames, extracts the specified color channel(s), and divides the ROI into grids.
	5. For each grid cell, detects motion or changes using contour analysis and marks cells as detected or not.
	6. Annotates the frame with colored rectangles (green for detected, red for not detected) and saves results to video and images.
	7. Tracks memory usage and execution time, and can export results to Excel.
- **Purpose:** Serves as the reference implementation for all improvements.

### `rparallel_he3.py`
- **Description:** Implements the base parallel algorithm for He3 video/image sequence processing using thread-based parallelism.
- **Processing Steps:**
	1. Loads configuration and supports both video files and image sequences.
	2. For each frame pair, applies histogram equalization to two ROIs in parallel using threads.
	3. Computes frame differences, extracts color channels, and divides each ROI into grids.
	4. Processes each grid cell in parallel using threads, performing contour analysis for detection.
	5. Annotates frames with colored rectangles for detected and non-detected cells in both ROIs.
	6. Tracks density values for each lane, memory usage, and execution time.
- **Purpose:** Serves as the reference parallel implementation for further improvements.

## Improved Scripts

### `roi_selector.py`
- **Description:** Interactive tool for selecting regions of interest (ROIs) in video, image, or image sequence.
- **Processing Steps:**
	1. Loads a video, image, or image sequence and displays the first frame.
	2. Allows the user to draw ROI rectangles using mouse interactions.
	3. Supports undo, reset, and saving ROI coordinates to a JSON file.
	4. Can step through frames/images for multi-frame ROI selection.
- **Improvement:** Enables precise, user-driven ROI selection for downstream processing scripts.
- **Base:** Can be used with both `seq.py` and `rparallel_he3.py` outputs.

### `rparallel_he3_mp.py`
- **Description:** Parallel He3 processing using Python's multiprocessing module for improved speed.
- **Processing Steps:**
	1. Loads configuration and input video/image sequence.
	2. Loads all frame pairs into memory, then processes them in parallel using multiple processes.
	3. For each frame pair, applies histogram equalization to ROIs, computes differences, extracts color channels, and divides ROIs into grids.
	4. Uses vectorized grid processing and multiprocessing to analyze each cell for motion/change detection.
	5. Annotates frames and writes output video, tracking density and performance statistics.
- **Improvement:** Utilizes true multiprocessing for faster computation and better CPU utilization compared to thread-based parallelism.
- **Base:** Improves upon `rparallel_he3.py`.

### `rparallel_he3_mpv.py`
- **Description:** Parallel He3 processing with memory and performance optimizations, using thread pools and vectorized operations.
- **Processing Steps:**
	1. Loads configuration and input video/image sequence.
	2. Applies histogram equalization to ROIs in parallel using threads.
	3. Uses vectorized operations for grid processing, reducing memory usage and improving speed.
	4. Annotates frames and writes output video, tracking density and performance statistics.
- **Improvement:** Optimizes memory usage and performance over the base parallel script by minimizing data copies and using efficient NumPy operations.
- **Base:** Improves upon `rparallel_he3.py`.

### `rparallel_he3_mpv_cont.py`
- **Description:** Parallel He3 processing with continuous (streaming) data handling and optimized vectorization.
- **Processing Steps:**
	1. Streams frame pairs from video or image sequence without loading all frames into memory.
	2. Applies in-place histogram equalization to ROIs for each frame pair.
	3. Uses highly optimized NumPy vectorization for grid processing, avoiding multiprocessing overhead.
	4. Annotates frames and writes output video, tracking density and performance statistics.
- **Improvement:** Adds support for continuous data streams, enabling real-time or memory-efficient analysis.
- **Base:** Improves upon `rparallel_he3_mpv.py`.

### `rparallel_he3_mpv_count_vec.py`
- **Description:** Parallel He3 processing with count-based vectorization for grid cell evaluation.
- **Processing Steps:**
	1. Streams frame pairs and applies in-place histogram equalization to ROIs.
	2. Uses vectorized pixel counting (instead of contours) to quickly evaluate grid cells for detection.
	3. Annotates frames and writes output video, tracking density and performance statistics.
- **Improvement:** Uses count vectorization for more efficient and faster grid cell evaluation, reducing computational overhead.
- **Base:** Improves upon `rparallel_he3_mpv.py`.

### `rparallel_he3_mpv_imp.py`
- **Description:** Parallel He3 processing with advanced implementation improvements for speed, accuracy, and memory efficiency.
- **Processing Steps:**
	1. Streams frame pairs in batches for efficient memory usage.
	2. Processes batches in parallel using multiprocessing, with ultra-fast vectorized grid evaluation.
	3. Uses pixel counting and minimal memory overhead for grid cell detection.
	4. Annotates frames and writes output video, tracking density and performance statistics.
- **Improvement:** Incorporates advanced algorithmic and implementation enhancements for speed, accuracy, and memory efficiency, including batch processing and optimized multiprocessing.
- **Base:** Improves upon `rparallel_he3_mpv.py` and related scripts.

## Usage

Each script can be run independently. For best results, start with the base scripts to understand the workflow, then use the improved versions for enhanced performance and features.

### How to Update the Video/Image Sequence Path

All scripts read their input path from the configuration file `user_input_data.json`. To change the video file or image sequence folder used for processing:

1. Open `user_input_data.json` in your editor.
2. Find the line with the key `"video"`:
	 ```json
	 {
		 "video": "path/to/your/video_or_image_folder",
		 ...existing config...
	 }
	 ```
3. Replace the value with the full path to your video file (e.g., `"D:/data/my_video.mp4"`) or image sequence folder (e.g., `"D:/data/images/"`).
4. Save the file.

All scripts will use this updated path the next time they are run.

---

For more details on each script, refer to the code comments and documentation within each file.