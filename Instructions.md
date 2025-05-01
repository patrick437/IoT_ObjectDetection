# Object Detection Using IMX500 - Instructions

This document provides instructions for using the object detection code with the Raspberry Pi IMX500 camera.

## Basic Setup

1. Save the Python script as `detector.py`
2. Ensure your model file is accessible (default: `traffic.rpk`)
3. If you have custom labels, place a `labels.txt` file in the same directory as your model

## Command Line Options

The script supports several command line arguments to control its behavior:

| Option | Description |
|--------|-------------|
| `--model PATH` | Path to the detection model (default: "traffic.rpk") |
| `--display` | Display video with detections in real-time |
| `--display_every N` | Display every N frames to reduce processing load (default: 10) |
| `--save_detections` | Save images with detection boxes |
| `--record_video` | Record video from images |
| `--fps N` | Frames per second for recorded videos (default: 30) |
| `--save_tensors` | Save tensor data for analysis |

## Usage Examples

### Basic Detection

To run basic object detection without any display or recording:

```bash
python detector.py --model /path/to/your/model.rpk
