import cv2
import numpy as np
import json
import os

class ROISelector:
    def __init__(self, source_path):
        self.source_path = source_path
        self.cap = None
        self.image_files = []
        self.current_image_index = 0
        self.is_image_sequence = False
        self.is_single_image = False
        self.roi_points = []
        self.current_roi = []
        self.roi_list = []
        self.drawing = False
        self.frame = None
        self.original_frame = None

        # Determine source type: video, single image or image folder
        if os.path.isdir(source_path):
            # load images from directory
            exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            files = [f for f in os.listdir(source_path) if f.lower().endswith(exts)]
            files.sort()
            self.image_files = [os.path.join(source_path, f) for f in files]
            if not self.image_files:
                print(f"Error: No images found in directory {source_path}")
                return
            self.is_image_sequence = True
            self.frame = cv2.imread(self.image_files[0])
            if self.frame is None:
                print(f"Error: Cannot read first image {self.image_files[0]}")
                return
            self.original_frame = self.frame.copy()
        elif os.path.isfile(source_path):
            # check if it's an image or video
            if source_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                self.is_single_image = True
                self.frame = cv2.imread(source_path)
                if self.frame is None:
                    print(f"Error: Cannot read image {source_path}")
                    return
                self.original_frame = self.frame.copy()
            else:
                # treat as video
                self.cap = cv2.VideoCapture(source_path)
                if not self.cap.isOpened():
                    print(f"Error: Cannot open video file {source_path}")
                    return
                ret, self.frame = self.cap.read()
                if not ret:
                    print("Error: Cannot read first frame")
                    return
                self.original_frame = self.frame.copy()
        else:
            print(f"Error: Source not found: {source_path}")
            return
        print("ROI Selector Instructions:")
        print("- Click and drag to select ROI rectangles")
        print("- Press 'r' to reset all ROIs")
        print("- Press 'u' to undo last ROI")
        print("- Press 's' to save ROI coordinates to file")
        print("- Press 'q' or ESC to quit")
    print("- Press SPACE to get next frame / next image")
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_roi = [(x, y)]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Draw temporary rectangle
                temp_frame = self.original_frame.copy()
                # Draw existing ROIs
                for i, roi in enumerate(self.roi_list):
                    cv2.rectangle(temp_frame, roi[0], roi[1], (0, 255, 0), 2)
                    cv2.putText(temp_frame, f"ROI {i+1}", 
                              (roi[0][0], roi[0][1]-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw current rectangle being drawn
                if len(self.current_roi) > 0:
                    cv2.rectangle(temp_frame, self.current_roi[0], (x, y), (255, 0, 0), 2)
                
                cv2.imshow('ROI Selector', temp_frame)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.current_roi.append((x, y))
                
                # Ensure rectangle has positive width and height
                x1, y1 = self.current_roi[0]
                x2, y2 = self.current_roi[1]
                
                # Swap coordinates if necessary
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                    
                # Add to ROI list
                self.roi_list.append([(x1, y1), (x2, y2)])
                print(f"ROI {len(self.roi_list)}: ({x1}, {y1}) to ({x2}, {y2}) - Width: {x2-x1}, Height: {y2-y1}")
                
                # Redraw frame with all ROIs
                self.draw_rois()
                
    def draw_rois(self):
        self.frame = self.original_frame.copy()
        for i, roi in enumerate(self.roi_list):
            cv2.rectangle(self.frame, roi[0], roi[1], (0, 255, 0), 2)
            cv2.putText(self.frame, f"ROI {i+1}", 
                       (roi[0][0], roi[0][1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display coordinates and dimensions
            x1, y1 = roi[0]
            x2, y2 = roi[1]
            width = x2 - x1
            height = y2 - y1
            
            cv2.putText(self.frame, f"({x1},{y1})", 
                       (roi[0][0], roi[1][1]+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(self.frame, f"{width}x{height}", 
                       (roi[0][0], roi[1][1]+40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('ROI Selector', self.frame)
        
    def next_frame(self):
        if self.is_image_sequence:
            self.current_image_index += 1
            if self.current_image_index >= len(self.image_files):
                print("End of image sequence")
                return
            frame = cv2.imread(self.image_files[self.current_image_index])
            if frame is None:
                print(f"Cannot read image {self.image_files[self.current_image_index]}")
                return
            self.original_frame = frame.copy()
            self.draw_rois()
            print(f"Moved to next image ({self.current_image_index+1}/{len(self.image_files)})")
        elif self.is_single_image:
            print("Single image mode: no next image")
        else:
            ret, frame = self.cap.read()
            if ret:
                self.original_frame = frame.copy()
                self.draw_rois()
                print("Moved to next frame")
            else:
                print("End of video or cannot read next frame")
            
    def reset_rois(self):
        self.roi_list = []
        self.frame = self.original_frame.copy()
        cv2.imshow('ROI Selector', self.frame)
        print("All ROIs reset")
        
    def undo_last_roi(self):
        if self.roi_list:
            removed = self.roi_list.pop()
            print(f"Removed ROI: {removed}")
            self.draw_rois()
        else:
            print("No ROIs to undo")
            
    def save_roi_coordinates(self):
        if not self.roi_list:
            print("No ROIs to save")
            return
            
        roi_data = {
            "video_file": self.video_path,
            "rois": []
        }
        
        for i, roi in enumerate(self.roi_list):
            x1, y1 = roi[0]
            x2, y2 = roi[1]
            roi_info = {
                f"roi_{i+1}": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "coordinates": f"({x1}, {y1}) to ({x2}, {y2})"
                }
            }
            roi_data["rois"].append(roi_info)
            
        # Save to JSON file
        with open("roi_coordinates.json", "w") as f:
            json.dump(roi_data, f, indent=4)
            
        print(f"ROI coordinates saved to 'roi_coordinates.json'")
        print("ROI Summary:")
        for i, roi in enumerate(self.roi_list):
            x1, y1 = roi[0]
            x2, y2 = roi[1]
            print(f"ROI {i+1}: x={x1}, y={y1}, width={x2-x1}, height={y2-y1}")
            
    def run(self):
        if self.frame is None:
            print("Error: No frame to display")
            return
            
        cv2.namedWindow('ROI Selector', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('ROI Selector', self.mouse_callback)
        
        cv2.imshow('ROI Selector', self.frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('r'):  # Reset
                self.reset_rois()
            elif key == ord('u'):  # Undo
                self.undo_last_roi()
            elif key == ord('s'):  # Save
                self.save_roi_coordinates()
            elif key == ord(' '):  # Space - next frame / image
                self.next_frame()
                
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()

def main():
    # You can change this path to your video file
    video_path = input("Enter video file path (or press Enter for default): ").strip()
    
    if not video_path:
        # Try to load from user_input_data.json
        try:
            with open("user_input_data.json", "r") as f:
                config = json.load(f)
                video_path = config["video"]
                print(f"Using video from config: {video_path}")
        except:
            print("No config file found. Please enter video path:")
            video_path = input("Video path: ").strip()
    
    if not video_path:
        print("No video path provided")
        return
        
    # Check if file exists
    import os
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
        
    roi_selector = ROISelector(video_path)
    roi_selector.run()

if __name__ == "__main__":
    main()