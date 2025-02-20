# This is a modified file of an official Raspberry PI example. Origin file: 
# https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py
# BSD 2-Clause License. For additional information read: LICENCE-Raspberry-PI

import sys
from functools import lru_cache
import cv2
import numpy as np
import time

from itkacher.date_utils import DateUtils
from itkacher.file_utils import FileUtils
from itkacher.video_recorder import VideoRecorder

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)


last_detections = []

threshold = 0.55
iou = 0.65
max_detections = 10

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP out."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]

    # Add tensor saving here
    try:
        timestamp = DateUtils.get_time()
        tensor_folder = f"./data/tensors/{DateUtils.get_date()}/"
        tensor_outputs = [boxes, scores, classes]
        
        # Create VideoRecorder instance (if not already created)
        video_recorder = VideoRecorder()
        video_recorder.save_tensor_data(tensor_outputs, timestamp, tensor_folder)
    except Exception as e:
        print(f"Error saving tensor data: {e}")

    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_tensors', action='store_true', help='Save tensor data')
    parser.add_argument('--record_video', action='store_true', help='Record video from images')
    args = parser.parse_args()

    model = "./imx500-models-backup/imx500_network_yolov8n_pp.rpk"

    # Initialize video recorder if needed
    video_recorder = VideoRecorder() if args.record_video else None

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(model)
    intrinsics = imx500.network_intrinsics

    # Your existing setup code...

    # Modify your main loop
    image_count = 0
    IMAGES_PER_VIDEO = 300  # Will create a 10-second video at 30fps

    while True:
        last_results = parse_detections(picam2.capture_metadata())
        
        # Record file to SD card
        data_folder = f"./data/images/{DateUtils.get_date()}/"
        try:
            # Save image
            current_time = DateUtils.get_time()
            image_path = f"{data_folder}/{current_time}.jpg"
            picam2.capture_file(image_path)
            image_count += 1

            # Save tensors if enabled
            if args.save_tensors and len(last_results) > 0:
                tensor_folder = f"./data/tensors/{DateUtils.get_date()}/"
                try:
                    tensor_outputs = [boxes, scores, classes]  # Get these from your parse_detections
                    video_recorder.save_tensor_data(tensor_outputs, current_time, tensor_folder)
                except Exception as error:
                    print(f"Error saving tensor data: {error}")

            # Create video if enough frames collected
            if args.record_video and image_count >= IMAGES_PER_VIDEO:
                try:
                    output_video = f"./data/videos/{DateUtils.get_date()}/video_{current_time}.mp4"
                    os.makedirs(os.path.dirname(output_video), exist_ok=True)
                    video_recorder.record_video(data_folder, output_video)
                    image_count = 0  # Reset counter
                except Exception as error:
                    print(f"Error creating video: {error}")

        except:
            FileUtils.create_folders(data_folder)

        if (len(last_results) > 0):
            for result in last_results:
                label = f"{labels[int(result.category)]} ({result.conf:.2f})"
                print(f"Detected {label}")
