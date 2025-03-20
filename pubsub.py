# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.
import sys
from functools import lru_cache
import cv2
import numpy as np
import time
import argparse
import os
import json
from datetime import datetime

from itkacher.date_utils import DateUtils
from itkacher.file_utils import FileUtils
from itkacher.video_recorder import VideoRecorder

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)
                                      
from awscrt import mqtt, http
from awsiot import mqtt_connection_builder
import sys
import threading
import time
import json
from utils.command_line_utils import CommandLineUtils

cmdData = CommandLineUtils.parse_sample_input_pubsub()
threshold = 0.55
iou = 0.65
max_detections = 10
received_count = 0
received_all_event = threading.Event()

def on_connection_interrupted(connection, error, **kwargs):
    print("Connection interrupted. error: {}".format(error))


# Callback when an interrupted connection is re-established.
def on_connection_resumed(connection, return_code, session_present, **kwargs):
    print("Connection resumed. return_code: {} session_present: {}".format(return_code, session_present))

    if return_code == mqtt.ConnectReturnCode.ACCEPTED and not session_present:
        print("Session did not persist. Resubscribing to existing topics...")
        resubscribe_future, _ = connection.resubscribe_existing_topics()

        # Cannot synchronously wait for resubscribe result because we're on the connection's event-loop thread,
        # evaluate result with a callback instead.
        resubscribe_future.add_done_callback(on_resubscribe_complete)


def on_resubscribe_complete(resubscribe_future):
    resubscribe_results = resubscribe_future.result()
    print("Resubscribe results: {}".format(resubscribe_results))

    for topic, qos in resubscribe_results['topics']:
        if qos is None:
            sys.exit("Server rejected resubscribe to topic: {}".format(topic))


# Callback when the subscribed topic receives a message
def on_message_received(topic, payload, dup, qos, retain, **kwargs):
    print("Received message from topic '{}': {}".format(topic, payload))
    global received_count
    received_count += 1
    if received_count == cmdData.input_count:
        received_all_event.set()

# Callback when the connection successfully connects
def on_connection_success(connection, callback_data):
    assert isinstance(callback_data, mqtt.OnConnectionSuccessData)
    print("Connection Successful with return code: {} session present: {}".format(callback_data.return_code, callback_data.session_present))

# Callback when a connection attempt fails
def on_connection_failure(connection, callback_data):
    assert isinstance(callback_data, mqtt.OnConnectionFailureData)
    print("Connection failed with error code: {}".format(callback_data.error))

# Callback when a connection has been disconnected or shutdown successfully
def on_connection_closed(connection, callback_data):
    print("Connection closed")

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
    return last_detections


@lru_cache
def get_labels():
    # Check if intrinsics is available and has labels
    if intrinsics is None:
        # Fallback: Read labels from a file if intrinsics is not available
        try:
            with open("/home/patrick/IoT_ObjectDetection/labels.txt", "r") as f:
                labels = [line.strip() for line in f.readlines()]
            return [label for label in labels if label and label != "-"]
        except FileNotFoundError:
            print("Warning: Neither intrinsics nor labels.txt found")
            return ["unknown"]  # Default fallback
    else:
        # Use intrinsics if available
        labels = intrinsics.labels
        if hasattr(intrinsics, 'ignore_dash_labels') and intrinsics.ignore_dash_labels:
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



if __name__ == '__main__':
    # Create the proxy options if the data is present in cmdData
    proxy_options = None
    if cmdData.input_proxy_host is not None and cmdData.input_proxy_port != 0:
        proxy_options = http.HttpProxyOptions(
            host_name=cmdData.input_proxy_host,
            port=cmdData.input_proxy_port)

    # Create a MQTT connection from the command line data
    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=cmdData.input_endpoint,
        port=cmdData.input_port,
        cert_filepath=cmdData.input_cert,
        pri_key_filepath=cmdData.input_key,
        ca_filepath=cmdData.input_ca,
        on_connection_interrupted=on_connection_interrupted,
        on_connection_resumed=on_connection_resumed,
        client_id=cmdData.input_clientId,
        clean_session=False,
        keep_alive_secs=30,
        http_proxy_options=proxy_options,
        on_connection_success=on_connection_success,
        on_connection_failure=on_connection_failure,
        on_connection_closed=on_connection_closed)

    if not cmdData.input_is_ci:
        print(f"Connecting to {cmdData.input_endpoint} with client ID '{cmdData.input_clientId}'...")
    else:
        print("Connecting to endpoint with client ID")
    connect_future = mqtt_connection.connect()

    # Future.result() waits until a result is available
    connect_future.result()
    print("Connected!")
    
    # Configure message topic from command line
    message_topic = cmdData.input_topic
    
    # Load the model
    model = "network.rpk"
    
    # This must be called before instantiation of Picamera2
    imx500 = IMX500(model)
    intrinsics = imx500.network_intrinsics

    # Initialize the camera
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={},
        buffer_count=12
    )

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)
    time.sleep(2)  # Allow the camera to warm up

    last_results = None
    picam2.pre_callback = draw_detections
    print("Started!")
    labels = get_labels()

    # Create necessary folders with proper permissions
    data_folder = f"./data/images/{DateUtils.get_date()}/"
    try:
        # Use sudo or run script as root to avoid permission issues
        FileUtils.create_folders(data_folder)
    except PermissionError:
        print(f"Permission error creating {data_folder}. Consider running with sudo.")
        # Try to continue without image saving
        data_folder = None

    # Main loop for object detection and publishing to AWS
    while True:
        # Capture metadata and parse detections
        last_results = parse_detections(picam2.capture_metadata())
        
        # Save image if folder was created successfully
        if data_folder:
            # Define image_path (this fixes the NameError)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_path = f"{data_folder}image_{timestamp}.jpg"
            try:
                picam2.capture_file(image_path)
            except Exception as e:
                print(f"Error saving image: {e}")

        if len(last_results) > 0:
            for result in last_results:
                label = f"{labels[int(result.category)]}"
                
                # Convert NumPy float32 to native Python float
                confidence = float(result.conf)
                confidence = round(confidence, 2)
                
                timestamp = datetime.now().isoformat()
                
                # Create JSON payload
                detection_payload = {
                    "timestamp": timestamp,
                    "object": label,
                    "confidence": confidence,
                    "device_id": cmdData.input_clientId
                }
                
                # Custom JSON encoder to handle NumPy types
                def convert_to_json_serializable(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return obj
                
                # Convert to JSON and publish
                detection_json = json.dumps(detection_payload, default=convert_to_json_serializable)
                print(f"Publishing detection to topic '{message_topic}': {detection_json}")
                
                try:
                    mqtt_connection.publish(
                        topic=message_topic,
                        payload=detection_json,
                        qos=mqtt.QoS.AT_LEAST_ONCE)
                except Exception as e:
                    print(f"Error publishing to AWS: {e}")
                
                # Print to console as well
                print(f"Detected {label} with confidence {confidence}")
                # Add a small delay to avoid maxing out CPU
                time.sleep(0.1)

    # Disconnect (this part will likely never be reached in this implementation)
    print("Disconnecting...")
    disconnect_future = mqtt_connection.disconnect()
    disconnect_future.result()
    print("Disconnected!")
