import sys
import argparse
from functools import lru_cache
import cv2
import numpy as np
import time
import os
import datetime

# Simple utility classes (instead of itkacher imports)
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
            print(f"Created folder: {folder_path}")

class VideoRecorder:
    """Simple implementation of VideoRecorder"""
    def __init__(self):
        self.frame_buffer = []
    
    def save_tensor_data(self, tensor_outputs, timestamp, tensor_folder):
        """Save tensor data to file"""
        try:
            FileUtils.create_folders(tensor_folder)
            np.save(f"{tensor_folder}/{timestamp}_boxes.npy", tensor_outputs[0])
            np.save(f"{tensor_folder}/{timestamp}_scores.npy", tensor_outputs[1])
            np.save(f"{tensor_folder}/{timestamp}_classes.npy", tensor_outputs[2])
            print(f"Saved tensor data to {tensor_folder}")
        except Exception as e:
            print(f"Error saving tensor data: {e}")
    
    def record_video(self, image_folder, output_video, fps=30):
        """Create a video from images in a folder"""
        try:
            # Check if the folder exists
            if not os.path.exists(image_folder):
                print(f"Image folder {image_folder} does not exist")
                return False
                
            # Get all jpg files in the folder
            images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
            if not images:
                print(f"No images found in {image_folder}")
                return False
                
            # Sort images by name
            images.sort()
            
            # Get the first image to determine video size
            first_image = cv2.imread(os.path.join(image_folder, images[0]))
            height, width, _ = first_image.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            # Add images to video
            for image in images:
                img = cv2.imread(os.path.join(image_folder, image))
                if img is not None:
                    video.write(img)
            
            # Release video writer
            video.release()
            print(f"Video created: {output_video}")
            return True
        except Exception as e:
            print(f"Error creating video: {e}")
            return False

# Import IMX500 Camera modules
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# Detection parameters
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
    
    # Get safely with default values
    bbox_normalization = getattr(intrinsics, 'bbox_normalization', False)
    
    # This part is important: np_outputs can be None, and that's OK
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        # Simply return the previous detections without error messages
        return last_detections
        
    input_w, input_h = imx500.get_input_size()
    
    # Process outputs based on model type
    if hasattr(intrinsics, 'postprocess') and intrinsics.postprocess == "nanodet":
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

    # Create detection objects for items above threshold
    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]

    # If you want to save tensor data, keep this
    if args.save_tensors and len(last_detections) > 0:
        try:
            timestamp = DateUtils.get_time()
            tensor_folder = f"./data/tensors/{DateUtils.get_date()}/"
            FileUtils.create_folders(tensor_folder)
            tensor_outputs = [boxes, scores, classes]
            
            if video_recorder:
                video_recorder.save_tensor_data(tensor_outputs, timestamp, tensor_folder)
        except Exception as e:
            print(f"Error saving tensor data: {e}")

    return last_detections


@lru_cache
def get_labels():
    """Get labels for detection categories"""
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

        # Draw ROI if available
        if hasattr(intrinsics, 'preserve_aspect_ratio') and intrinsics.preserve_aspect_ratio:
            try:
                b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
                color = (255, 0, 0)  # red
                cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))
            except Exception as e:
                print(f"Error drawing ROI: {e}")


def draw_detections_on_frame(frame, detections):
    """Draw the detections on a frame and return the modified frame"""
    if not detections:
        return frame
        
    labels = get_labels()
    frame_copy = frame.copy()
    
    for detection in detections:
        x, y, w, h = detection.box
        label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x + 5
        text_y = y + 15

        # Draw the background rectangle
        cv2.rectangle(frame_copy,
                     (text_x, text_y - text_height),
                     (text_x + text_width, text_y + baseline),
                     (255, 255, 255),  # Background color (white)
                     cv2.FILLED)

        # Draw text on top of the background
        cv2.putText(frame_copy, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw detection box
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    
    return frame_copy


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_tensors', action='store_true', help='Save tensor data')
    parser.add_argument('--record_video', action='store_true', help='Record video from images')
    parser.add_argument('--display', action='store_true', help='Display video with detections')
    parser.add_argument('--model', type=str, default="./imx500-models-backup/imx500_network_yolov8n_pp.rpk",
                       help='Path to model file')
    parser.add_argument('--video_delay', type=float, default=0.01,
                       help='Delay between frames in seconds (default: 0.01)')
    args = parser.parse_args()

    # Set model path from arguments or use default
    model = args.model

    # Initialize video recorder if needed
    video_recorder = VideoRecorder() if args.record_video else None

    print(f"Using model: {model}")
    print(f"Model exists: {os.path.exists(model)}")

    # This must be called before instantiation of Picamera2
    try:
        imx500 = IMX500(model)
        intrinsics = imx500.network_intrinsics
        print("IMX500 initialized successfully")
    except Exception as e:
        print(f"Error initializing IMX500: {e}")
        sys.exit(1)

    # Initialize the Picamera2 object
    picam2 = Picamera2()
    
    # Configure the camera with proper error handling
    try:
        # Get the input size for the camera configuration
        input_size = imx500.get_input_size()
        print(f"Camera input size: {input_size}")
        
        # Check if get_transform exists
        transform = None
        if hasattr(imx500, 'get_transform'):
            transform = imx500.get_transform()
        
        # Create the camera configuration
        camera_config = picam2.create_preview_configuration(
            main={"size": input_size},
            transform=transform,
            buffer_count=4
        )
        picam2.configure(camera_config)
        
        # Set up camera metadata
        if hasattr(imx500, 'post_callback'):
            picam2.post_callback = imx500.post_callback
        else:
            print("Warning: imx500.post_callback not found, skipping this step")
    except Exception as e:
        print(f"Error configuring camera: {e}")
        # Fallback configuration if the specialized configuration fails
        print("Attempting to use default camera configuration...")
        default_config = picam2.create_preview_configuration()
        picam2.configure(default_config)
        print("Using default camera configuration.")
    
    # Start the camera
    picam2.start()
    
    # Allow time for camera to initialize
    print("Waiting for camera to initialize...")
    time.sleep(2)
    
    # Get the labels
    labels = get_labels()
    print(f"Using label set: {labels}")

    # Main loop variables
    image_count = 0
    IMAGES_PER_VIDEO = 300  # Will create a 10-second video at 30fps

    try:
        print("Starting object detection...")
        while True:
            # Get detections - if this returns None, it's ok
            last_results = parse_detections(picam2.capture_metadata())
            
            # Capture frame for display/saving
            if args.display or args.record_video:
                frame = picam2.capture_array()
                if args.display:
                    annotated_frame = draw_detections_on_frame(frame, last_results)
                    cv2.imshow("Object Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Record file to SD card
            data_folder = f"./data/images/{DateUtils.get_date()}/"
            try:
                # Ensure the folder exists
                FileUtils.create_folders(data_folder)
                
                # Save image
                current_time = DateUtils.get_time()
                image_path = f"{data_folder}/{current_time}.jpg"
                
                if args.record_video and args.display:
                    # Save the annotated frame
                    cv2.imwrite(image_path, annotated_frame)
                else:
                    # Save the raw camera frame
                    picam2.capture_file(image_path)
                    
                image_count += 1

                # Create video if enough frames collected
                if args.record_video and image_count >= IMAGES_PER_VIDEO and video_recorder:
                    try:
                        video_folder = f"./data/videos/{DateUtils.get_date()}/"
                        FileUtils.create_folders(video_folder)
                        output_video = f"{video_folder}/video_{current_time}.mp4"
                        video_recorder.record_video(data_folder, output_video)
                        image_count = 0  # Reset counter
                        
                        # Optionally clean up the images folder after making the video
                        for file in os.listdir(data_folder):
                            if file.endswith('.jpg'):
                                os.remove(os.path.join(data_folder, file))
                    except Exception as error:
                        print(f"Error creating video: {error}")

            except Exception as e:
                print(f"Error in main loop: {e}")
                FileUtils.create_folders(data_folder)

            # Print detected objects
            if last_results and len(last_results) > 0:
                for result in last_results:
                    label = f"{labels[int(result.category)]} ({result.conf:.2f})"
                    print(f"Detected {label}")
            
            # Add delay to reduce CPU usage
            time.sleep(args.video_delay)
            
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        # Clean up
        if args.display:
            cv2.destroyAllWindows()
        picam2.stop()
        print("Camera stopped and resources released")
