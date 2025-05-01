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
        self.saved_images = []
    
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
    
    def process_image(self, image_path):
        """Add image to the list of saved images"""
        self.saved_images.append(image_path)
        
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

def warm_up_camera():
    """Warm up the camera and model by capturing a few frames"""
    print("Warming up camera and model...")
    for i in range(3):  # Capture 3 warm-up frames
        frame = picam2.capture_array()
        metadata = picam2.capture_metadata()
        # Try to get outputs but don't process them yet
        outputs = imx500.get_outputs(metadata, add_batch=True)
        print(f"Warm-up frame {i+1}: {'outputs received' if outputs is not None else 'no outputs'}")
        time.sleep(0.5)
    print("Warm-up complete")

def parse_detections(metadata: dict):
    """Parse the output tensor into detected objects."""
    global last_detections
    
    # Get model outputs - this can return None and that's OK
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        # Simply return previous detections without printing error message
        return last_detections
    
    input_w, input_h = imx500.get_input_size()
    
    # Get safely with default values
    bbox_normalization = getattr(intrinsics, 'bbox_normalization', False)
    
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
        boxes = list(zip(*boxes))  # Convert to list to avoid depletion

    # Create detection objects
    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    
    if last_detections:
        print(f"Detected {len(last_detections)} objects")
    
    # Save tensor data if enabled
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
    """Get model labels with better error handling"""
    global intrinsics
    
    if intrinsics is None:
        print("WARNING: intrinsics is None, returning default labels")
        return ["object"]  # Default label
        
    if not hasattr(intrinsics, 'labels'):
        print("WARNING: intrinsics has no 'labels' attribute, returning default labels")
        return ["object"]  # Default label
    
    labels = intrinsics.labels
    if hasattr(intrinsics, 'ignore_dash_labels') and intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    
    return labels

def draw_detections_on_frame(frame, detections):
    """Draw the detections on the frame and return the modified frame"""
    if detections is None or len(detections) == 0:
        return frame
    
    labels = get_labels()
    frame_copy = frame.copy()
    
    for detection in detections:
        try:
            # Extract detection data with debug information
            x, y, w, h = detection.box
            print(f"Drawing box: x={x}, y={y}, w={w}, h={h}")
            
            # Make sure coordinates are valid
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                print(f"Warning: Invalid box coordinates: {detection.box}")
                continue
                
            # Ensure values are integers
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Get label text with safety checks
            try:
                label_idx = int(detection.category)
                if 0 <= label_idx < len(labels):
                    label_text = labels[label_idx]
                else:
                    print(f"Warning: Invalid label index {label_idx}, max is {len(labels)-1}")
                    label_text = f"Class_{label_idx}"
            except Exception as e:
                print(f"Error getting label: {e}")
                label_text = "Unknown"
                
            label = f"{label_text} ({detection.conf:.2f})"
            print(f"Drawing label: {label}")

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

            # Draw detection box - use green color with thickness 2
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        except Exception as e:
            print(f"Error drawing detection: {e}")
    
    return frame_copy


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_tensors', action='store_true', help='Save tensor data')
    parser.add_argument('--record_video', action='store_true', help='Record video from images')
    parser.add_argument('--model', type=str, default="traffic.rpk", 
                       help='Path to the detection model')
    parser.add_argument('--display', action='store_true', help='Display video with detections using cv2.imshow')
    parser.add_argument('--display_every', type=int, default=10, 
                       help='Display every N frames to reduce processing load (default: 10)')
    parser.add_argument('--save_detections', action='store_true', help='Save images with detection boxes')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for recorded videos')
    args = parser.parse_args()

    # Set the model path - you can override with --model argument
    model = args.model if args.model else "traffic.rpk"
    
    print(f"Using model: {model}")
    print(f"Model exists: {os.path.exists(model)}")
    
    # Initialize IMX500 with the model
    imx500 = IMX500(model)
    intrinsics = imx500.network_intrinsics
    
    # Create fallback intrinsics if needed
    if intrinsics is None:
        from picamera2.devices.imx500 import NetworkIntrinsics
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
        intrinsics.bbox_normalization = False
        intrinsics.labels = ["object"]  # Default label
        intrinsics.ignore_dash_labels = False
        intrinsics.bbox_order = "yx"
        print("Created fallback intrinsics")
    
    # Check for labels.txt in the model directory
    try:
        model_dir = os.path.dirname(model) if os.path.dirname(model) else "."
        labels_path = os.path.join(model_dir, "labels.txt")
        
        if os.path.exists(labels_path):
            print(f"Found labels file: {labels_path}")
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            
            # Set labels in intrinsics
            intrinsics.labels = labels
            print(f"Loaded labels: {labels}")
        else:
            print(f"No labels.txt found at {labels_path}, using model's built-in labels")
            # The built-in labels are already in the intrinsics for standard models
            if hasattr(intrinsics, 'labels') and intrinsics.labels:
                print(f"Model has {len(intrinsics.labels)} built-in labels")
                if len(intrinsics.labels) < 10:
                    print(f"Labels: {intrinsics.labels}")
                else:
                    print(f"First few labels: {intrinsics.labels[:5]}...")
    except Exception as e:
        print(f"Error handling labels: {e}")
        # No need to set default labels - the intrinsics already has them
    
    # Initialize the camera
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls = {},
        buffer_count=12
    )

    # Show network firmware progress bar if available
    if hasattr(imx500, 'show_network_fw_progress_bar'):
        imx500.show_network_fw_progress_bar()
    
    # Start the camera
    picam2.start(config, show_preview=False)
    
    # Warm up the camera and model
    warm_up_camera()
    
    # Initialize video recorder if needed
    video_recorder = VideoRecorder() if args.record_video else None
    
    # Calculate frames per video based on duration
    IMAGES_PER_VIDEO = 300  # Will create a 10-second video at 30fps
    
    # Initialize frame counter for display
    frame_count = 0
    
    try:
        print("Starting object detection...")
        while True:
            # Capture frame
            frame = picam2.capture_array()
            frame_count += 1
            
            # Capture and parse detections
            metadata = picam2.capture_metadata()
            last_results = parse_detections(metadata)
            
            # Display frame if needed
            if args.display and frame_count % args.display_every == 0:
                display_frame = draw_detections_on_frame(frame, last_results)
                cv2.imshow("Object Detection", display_frame)
                
                # Break loop if 'q' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    print("Display closed by user")
                    break
            
            # Print detected objects
            if last_results and len(last_results) > 0:
                labels = get_labels()
                for result in last_results:
                    try:
                        label_idx = int(result.category)
                        if 0 <= label_idx < len(labels):
                            label_text = labels[label_idx]
                        else:
                            label_text = "Unknown"
                        
                        confidence = result.conf
                        print(f"Detected {label_text} with confidence {confidence:.2f}")
                    except Exception as e:
                        print(f"Error printing detection: {e}")
            
            # Record image to SD card
            data_folder = f"./data/images/{DateUtils.get_date()}/"
            try:
                # Ensure the folder exists
                FileUtils.create_folders(data_folder)
                
                # Save image
                current_time = DateUtils.get_time()
                image_path = f"{data_folder}/{current_time}.jpg"
                
                if args.display:
                    # Save the annotated frame
                    annotated_frame = draw_detections_on_frame(frame, last_results)
                    cv2.imwrite(image_path, annotated_frame)
                else:
                    # Save the raw camera frame
                    picam2.capture_file(image_path)
                
                # Process the image for video if recording is enabled
                if args.record_video and video_recorder:
                    video_recorder.process_image(image_path)
                    
                    # Create video if enough frames collected
                    if len(video_recorder.saved_images) >= IMAGES_PER_VIDEO:
                        try:
                            video_folder = f"./data/videos/{DateUtils.get_date()}/"
                            FileUtils.create_folders(video_folder)
                            output_video = f"{video_folder}/video_{current_time}.mp4"
                            video_recorder.record_video(data_folder, output_video, args.fps)
                            
                            # Reset saved images
                            video_recorder.saved_images = []
                            
                            # Optionally clear the images folder
                            for file in os.listdir(data_folder):
                                try:
                                    if file.endswith('.jpg'):
                                        os.remove(os.path.join(data_folder, file))
                                except Exception as e:
                                    print(f"Error deleting file: {e}")
                        except Exception as e:
                            print(f"Error creating video: {e}")

            except Exception as e:
                print(f"Error in main loop: {e}")
                FileUtils.create_folders(data_folder)

            # Save detections if enabled
            if args.save_detections and last_results:
                # Create folder if it doesn't exist
                detections_folder = f"./data/detections/{DateUtils.get_date()}/"
                FileUtils.create_folders(detections_folder)
                
                # Save the annotated frame
                annotated_frame = draw_detections_on_frame(frame, last_results)
                detection_path = f"{detections_folder}/{DateUtils.get_time()}_annotated.jpg"
                cv2.imwrite(detection_path, annotated_frame)
                print(f"Saved annotated frame to {detection_path}")
            
            # Delay between frames
            time.sleep(0.1)  # Small delay to reduce CPU usage
            
    except KeyboardInterrupt:
        print("Program terminated by user")
        cv2.destroyAllWindows()  # Close any open windows
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()  # Close any open windows
    finally:
        # Clean up
        try:
            picam2.stop()
            print("Camera stopped and resources released")
        except:
            pass
