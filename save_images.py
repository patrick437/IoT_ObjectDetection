import sys
import os
import time
import argparse
import cv2
import numpy as np
from functools import lru_cache
import datetime

# Import IMX500 Camera modules
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# --- Configurable Settings ---
SAVE_EVERY_N_FRAMES = 5  # Save annotated image every 5 frames
MAX_SAVED_IMAGES = 100   # Auto-delete oldest if folder exceeds this
DISPLAY_EVERY = 10       # Only update display every 10 frames (reduces CPU load)
THRESHOLD = 0.55         # Confidence threshold for detections
# ----------------------------

# Simple utility classes
class DateUtils:
    @staticmethod
    def get_date():
        """Get current date in YYYY-MM-DD format"""
        return datetime.datetime.now().strftime("%Y-%m-%d")
    
    @staticmethod
    def get_time():
        """Get current time in HHMMSS format"""
        return datetime.datetime.now().strftime("%H%M%S")

class FileUtils:
    @staticmethod
    def create_folders(folder_path):
        """Create folder if it doesn't exist"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

# Detection parameters
last_detections = []
iou = 0.65
max_detections = 10

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def warm_up_camera():
    """Warm up the camera and model by capturing a few frames"""
    print("Warming up camera...")
    for _ in range(3):
        frame = picam2.capture_array()
        metadata = picam2.capture_metadata()
        _ = imx500.get_outputs(metadata, add_batch=True)
        time.sleep(0.5)
    print("Warm-up complete")

def parse_detections(metadata: dict):
    """Parse the output tensor into detected objects."""
    global last_detections, intrinsics
    
    if intrinsics is None:
        return last_detections
    
    # Get model outputs
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return last_detections
    
    input_w, input_h = imx500.get_input_size()
    
    try:
        if hasattr(intrinsics, 'postprocess') and intrinsics.postprocess == "nanodet":
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0], conf=THRESHOLD, iou_thres=iou, max_out_dets=max_detections
            )[0]
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if getattr(intrinsics, 'bbox_normalization', False):
                boxes = boxes / input_h
            if getattr(intrinsics, 'bbox_order', 'yx') == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]

        # Filter by confidence threshold
        last_detections = [
            Detection(box, category, score, metadata)
            for box, score, category in zip(boxes, scores, classes)
            if score > THRESHOLD
        ]
        return last_detections
    
    except Exception as e:
        print(f"Detection error: {e}")
        return last_detections

@lru_cache
def get_labels():
    """Get model labels with fallback to 'object' if none exist."""
    if intrinsics is None or not hasattr(intrinsics, 'labels'):
        return ["object"]
    return [label for label in intrinsics.labels if label != "-"]

def draw_detections_on_frame(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    if not detections:
        return frame
    
    labels = get_labels()
    frame_copy = frame.copy()
    
    for detection in detections:
        x, y, w, h = detection.box
        label_idx = int(detection.category)
        label_text = labels[label_idx] if 0 <= label_idx < len(labels) else "Unknown"
        label = f"{label_text} ({detection.conf:.2f})"
        
        # Draw bounding box
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_copy, (x, y - text_h - 5), (x + text_w, y), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame_copy, label, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame_copy

def cleanup_old_files(folder, max_files):
    """Delete oldest files if folder exceeds max_files."""
    files = sorted(os.listdir(folder), key=lambda f: os.path.getctime(os.path.join(folder, f)))
    while len(files) > max_files:
        os.remove(os.path.join(folder, files[0]))
        files.pop(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="traffic.rpk", help='Path to model file')
    parser.add_argument('--display', action='store_true', help='Enable live display')
    args = parser.parse_args()

    # Initialize IMX500
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    
    # Initialize camera
    picam2 = Picamera2(imx500.camera_num)
    picam2.start(picam2.create_preview_configuration(buffer_count=12))
    warm_up_camera()

    try:
        frame_count = 0
        print("Running detection (Press Ctrl+C to stop)...")
        
        while True:
            frame = picam2.capture_array()
            metadata = picam2.capture_metadata()
            detections = parse_detections(metadata)
            frame_count += 1

            # Display (optional)
            if args.display and frame_count % DISPLAY_EVERY == 0:
                cv2.imshow("Detections", draw_detections_on_frame(frame, detections))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save annotated image periodically
            if frame_count % SAVE_EVERY_N_FRAMES == 0 and detections:
                date_folder = f"./detections/{DateUtils.get_date()}/"
                FileUtils.create_folders(date_folder)
                img_path = f"{date_folder}/{DateUtils.get_time()}.jpg"
                cv2.imwrite(img_path, draw_detections_on_frame(frame, detections))
                
                # Auto-cleanup if folder too large
                if len(os.listdir(date_folder)) > MAX_SAVED_IMAGES:
                    cleanup_old_files(date_folder, MAX_SAVED_IMAGES)

            time.sleep(0.1)  # Reduce CPU usage

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
