import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def read_and_analyze_tensors(folder_path):
    """Read and analyze all tensor files in the given folder"""
    print(f"Reading tensors from: {folder_path}")
    
    # Get all .npz files in the folder
    tensor_files = list(Path(folder_path).glob('**/*_tensor.npz'))
    if not tensor_files:
        print("No tensor files found!")
        return
        
    print(f"Found {len(tensor_files)} tensor files")
    
    for tensor_file in sorted(tensor_files):
        print(f"\nAnalyzing: {tensor_file.name}")
        try: 
            data = np.load(tensor_file, allow_pickle=True)
            
            # Extract data
            boxes = data['boxes']
            scores = data['scores']
            classes = data['classes']
            
            # Print analysis
            print(f"Number of detections: {len(scores)}")
            if len(scores) > 0:
                print(f"Confidence scores: min={scores.min():.2f}, max={scores.max():.2f}, avg={scores.mean():.2f}")
                print("Detected classes:", np.unique(classes))
                print("Bounding boxes shape:", boxes.shape)
        except Exception as e:
            print(f"Error while reading file {tensor_file.name}: {e}")
    
    try:
        # Plot summary
        plt.figure(figsize=(10, 6))
        plt.title("Detections per File")
        allow_pickle=True
        plt.bar(range(len(tensor_files)), [np.load(f)['scores'].shape[0] for f in tensor_files])
        plt.xlabel("File Number")
        plt.ylabel("Number of Detections")
        plt.show()
    except Exception as e:
        print(f"Error creating plot: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "./data/tensors"  # default folder
    
    read_and_analyze_tensors(folder)
