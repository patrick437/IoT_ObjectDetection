import cv2
import os
import numpy as np
import json
from datetime import datetime

class VideoRecorder:

    def save_tensor_data(self, tensor_outputs, timestamp, tensor_folder):
        """Save tensor outputs and detection data"""
        # Create folder for tensor data if it doesn't exist
        os.makedirs(tensor_folder, exist_ok=True)
        
        # Save raw tensor outputs
        tensor_file = os.path.join(tensor_folder, f"{timestamp}_tensor.npz")
        np.savez_compressed(
            tensor_file,
            boxes=tensor_outputs[0],
            scores=tensor_outputs[1],
            classes=tensor_outputs[2]
        )
    
    def record_video(self, input_folder: str, output_video_file: str):
        images = [img for img in os.listdir(input_folder) if img.endswith(".jpg") or img.endswith(".png")]
        images.sort()  

        first_image_path = os.path.join(input_folder, images[0])
        frame = cv2.imread(first_image_path)
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        video = cv2.VideoWriter(output_video_file, fourcc, 30.0, (width, height))  # 30.0 is the FPS

        for image in images:
            image_path = os.path.join(input_folder, image)
            frame = cv2.imread(image_path)
            video.write(frame)

        video.release()
