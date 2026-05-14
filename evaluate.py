import math
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
from scipy.optimize import linear_sum_assignment

ANNOTATIONS_CSV = 'data/annotations.csv'


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
        '''
        write predictions on image and send to output_path
        '''
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


def evaluate_entire_dataset(detector, csv_path, img_dir, distance_threshold=20.0):
    """
    Evaluates the model across the entire dataset calculating Precision, 
    Recall, F1-Score, and Mean Angle Error.
    """
    out_dir = 'output'
    img_dir = 'data/images'
    csv_path = 'data/annotations.csv'
    df = pd.read_csv(csv_path)
    unique_imgs = df['image'].unique()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    angle_errors = []
    center_errors = []

    print(f"Evaluating {len(unique_imgs)} images...")
    
    for img_name in unique_imgs:
        img_path = f"{img_dir}/{img_name}"
        
        # Get Ground Truth
        gt_subset = df[df['image'] == img_name]
        gt_centers = gt_subset[['center_x', 'center_y']].values
        gt_angles = gt_subset['angle_deg'].values
        
        # Get Predictions
        preds = detector.predict(img_path)
                
        # send prdictions to output
        for i, p in enumerate(preds):
            print(f"Tube {i+1}: Center=({p['center_x']:.1f}, {p['center_y']:.1f}), Angle={p['angle_deg']:.1f}°")
            
        detector.visualize(img_path, preds, output_path=os.path.join(out_dir,img_name))

        if len(preds) == 0:
            total_fn += len(gt_centers)
            continue
            
        pred_centers = np.array([[p['center_x'], p['center_y']] for p in preds])
        pred_angles = np.array([p['angle_deg'] for p in preds])
        
        if len(gt_centers) == 0:
            total_fp += len(preds)
            continue
            
        # Create a Cost Matrix (Euclidean distance between every GT and Prediction)
        cost_matrix = np.zeros((len(gt_centers), len(pred_centers)))
        for i, gt_c in enumerate(gt_centers):
            for j, pr_c in enumerate(pred_centers):
                cost_matrix[i, j] = math.hypot(gt_c[0] - pr_c[0], gt_c[1] - pr_c[1])
                
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_gt = set()
        matched_pred = set()
        
        # Classify matches as TP based on distance threshold
        for gt_idx, pred_idx in zip(row_ind, col_ind):
            dist = cost_matrix[gt_idx, pred_idx]
            
            if dist <= distance_threshold:

                total_tp += 1
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                

                true_a = gt_angles[gt_idx]
                pred_a = pred_angles[pred_idx]
                diff = abs(true_a - pred_a)
                ang_err = min(diff, 360 - diff)
                angle_errors.append(ang_err)
                center_errors.append(dist)

        total_fp += len(pred_centers) - len(matched_pred)

        total_fn += len(gt_centers) - len(matched_gt)
        
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_angle_err = np.mean(angle_errors) if len(angle_errors) > 0 else 0.0
    mean_centers_err = np.mean(center_errors) if len(center_errors) > 0 else 0.0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Mean_Angle_Error': mean_angle_err,
        'Total_TP': total_tp,
        'Total_FP': total_fp,
        'Total_FN': total_fn,
        'Mean_Centers_Error': mean_centers_err
    }


if __name__ == "__main__":
    detector = TubePoseDetector()
    csv_path = 'data/annotations.csv'
    img_dir = 'data/images'

    # Threshold is 20 pixels (roughly half the width of a tube lid)
    results = evaluate_entire_dataset(detector, csv_path, img_dir, distance_threshold=20.0)
    
    print("\n" + "="*40)
    print("      FINAL DATASET EVALUATION")
    print("="*40)
    print(f"Total True Positives:   {results['Total_TP']}")
    print(f"Total False Positives:  {results['Total_FP']}")
    print(f"Total False Negatives:  {results['Total_FN']}")
    print("-" * 40)
    print(f"Precision:         {results['Precision']:.4f}")
    print(f"Recall:            {results['Recall']:.4f}")
    print(f"F1-Score:          {results['F1_Score']:.4f}")
    print("-" * 40)
    print(f"Mean Angle Error:      {results['Mean_Angle_Error']:.2f} degrees")
    print(f"Mean Centers Error:    {results['Mean_Centers_Error']:.2f} px")
    print("="*40)