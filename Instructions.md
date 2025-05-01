Object Detection Using IMX500 - Instructions
This document provides instructions for using the object detection code with the Raspberry Pi IMX500 camera.
Basic Setup

Save the Python script as detector.py
Ensure your model file is accessible (default: traffic.rpk)
If you have custom labels, place a labels.txt file in the same directory as your model

Command Line Options
The script supports several command line arguments to control its behavior:
OptionDescription--model PATHPath to the detection model (default: "traffic.rpk")--displayDisplay video with detections in real-time--display_every NDisplay every N frames to reduce processing load (default: 10)--save_detectionsSave images with detection boxes--record_videoRecord video from images--fps NFrames per second for recorded videos (default: 30)--save_tensorsSave tensor data for analysis
Usage Examples
Basic Detection
To run basic object detection without any display or recording:
bashpython detector.py --model /path/to/your/model.rpk
This will run the detection and print information about detected objects to the console.
Display Detection Results
To see the detection results in real-time:
bashpython detector.py --display
Press 'q' in the display window to exit.
Save Detection Images
To save images with detection boxes drawn on them:
bashpython detector.py --save_detections
Images will be saved to ./data/detections/YYYY-MM-DD/.
Record Video
To record 10-second video clips of the detections:
bashpython detector.py --record_video
Videos will be saved to ./data/videos/YYYY-MM-DD/.
Combined Options
You can combine multiple options:
bashpython detector.py --model custom_model.rpk --display --record_video --fps 24
This will use your custom model, display detections in real-time, and record videos at 24 fps.
Custom Labels
For custom models, create a labels.txt file in the same directory as your model with one label per line:
car
truck
bus
motorcycle
pedestrian
If no labels.txt file is found, the script will use the labels embedded in the model (if available).
Output Folders
The script creates the following folder structure:

./data/images/YYYY-MM-DD/ - Raw captured images
./data/videos/YYYY-MM-DD/ - Recorded videos
./data/detections/YYYY-MM-DD/ - Images with detection boxes
./data/tensors/YYYY-MM-DD/ - Tensor data (if enabled)

Troubleshooting

"No outputs received from model" - This is normal. The IMX500 may not provide outputs for every frame.
Camera not found - Make sure the IMX500 camera is properly connected and recognized.
Model not found - Double-check the path to your model file.
No detections - Try adjusting the detection threshold (default: 0.55).

Exiting the Program
To exit the program, press Ctrl+C in the terminal or press 'q' in the display window (if --display is enabled).
