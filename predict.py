import math
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
from scipy.optimize import linear_sum_assignment


class TubePoseDetector:
    def __init__(self, model_path='runs/pose/train/weights/best.pt'):
        print(f"Loading Pose model from {model_path}...")
        self.model = YOLO(model_path)

    def predict(self, image_path, conf_threshold=0.5):
        results = self.model(image_path, conf=conf_threshold, verbose=False)[0]
        predictions = []
        
        # Check if keypoints were detected
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            # xy contains the coordinates: shape is (Number of tubes, 2 keypoints, 2 coords)
            keypoints_batch = results.keypoints.xy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            
            for kpts, conf in zip(keypoints_batch, confidences):
                kp1_x, kp1_y = kpts[0] # Center
                kp2_x, kp2_y = kpts[1] # Tab
                
                # Calculate angle using atan2
                # dy is flipped (kp1_y - kp2_y) because image Y-coordinates go DOWN, 
                # but standard math Y-coordinates go UP.
                dx = kp2_x - kp1_x
                dy = kp1_y - kp2_y 
                
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad) % 360
                
                predictions.append({
                    'center_x': float(kp1_x),
                    'center_y': float(kp1_y),
                    'tab_x': float(kp2_x),
                    'tab_y': float(kp2_y),
                    'angle_deg': float(angle_deg),
                    'confidence': float(conf)
                })
                
        return predictions

    def visualize(self, image_path, predictions,output_path):
        img = cv2.imread(image_path)
        
        for pred in predictions:
            cx, cy = int(pred['center_x']), int(pred['center_y'])
            tx, ty = int(pred['tab_x']), int(pred['tab_y'])
            
            # Draw Center (Red)
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
            # Draw Tab (Blue)
            cv2.circle(img, (tx, ty), 4, (255, 0, 0), -1)
            
            # Draw the vector line connecting them (Green)
            cv2.line(img, (cx, cy), (tx, ty), (0, 255, 0), 2)
            
            # Label angle
            cv2.putText(img, f"{pred['angle_deg']:.0f} deg", (cx + 10, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    # Define your directories
    in_dir = "pred_in"
    out_dir = "pred_out"
    detector = TubePoseDetector()
    # Safely create the output directory if it doesn't exist yet
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Looking for images in '{in_dir}'...")
    
    # Loop through every file in the input directory
    for filename in os.listdir(in_dir):
        # Only process files that end with .png
        if filename.lower().endswith('.png'):
            # Safely build the full path to the input image
            img_path = os.path.join(in_dir, filename)
            
            print(f"\n--- Processing: {filename} ---")
            
            # 1. Predict
            preds = detector.predict(img_path)
            
            # 2. Print results to terminal
            for i, p in enumerate(preds):
                print(f" Tube {i+1}: Center=({p['center_x']:.1f}, {p['center_y']:.1f}), Angle={p['angle_deg']:.1f}°")
            
            # 3. Visualize and save
            # Build the full path for the output image (e.g., "preds_out/image1.png")
            out_path = os.path.join(out_dir, filename)
            detector.visualize(img_path, preds, output_path=out_path)
            
    print(f"\nBatch processing complete! Check the '{out_dir}' folder.")