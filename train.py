import os
import shutil
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

RAW_IMAGE_DIR = 'data/images'
ANNOTATIONS_CSV = 'data/annotations.csv'
YOLO_DATASET_DIR = 'yolo_pose_dataset' 

def setup_yolo_directories():
    dirs = [
        f'{YOLO_DATASET_DIR}/images/train', f'{YOLO_DATASET_DIR}/images/val',
        f'{YOLO_DATASET_DIR}/labels/train', f'{YOLO_DATASET_DIR}/labels/val'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def convert_and_split_data(test_size=0.15, img_w=640, img_h=480):
    df = pd.read_csv(ANNOTATIONS_CSV)
    unique_images = df['image'].unique()
    train_imgs, val_imgs = train_test_split(unique_images, test_size=test_size, random_state=42)
    
    def process_split(images, split_name):
        for image_name in images:
            shutil.copy(os.path.join(RAW_IMAGE_DIR, image_name), 
                        os.path.join(YOLO_DATASET_DIR, 'images', split_name, image_name))
            
            txt_path = os.path.join(YOLO_DATASET_DIR, 'labels', split_name, image_name.replace('.png', '.txt'))
            image_annotations = df[df['image'] == image_name]
            
            with open(txt_path, 'w') as f:
                for _, row in image_annotations.iterrows():
                    # Standard Bounding Box (Center format, normalized)
                    bbox_cx = (row['bbox_x'] + row['bbox_w'] / 2) / img_w
                    bbox_cy = (row['bbox_y'] + row['bbox_h'] / 2) / img_h
                    norm_w = row['bbox_w'] / img_w
                    norm_h = row['bbox_h'] / img_h
                    
                    # Keypoint 1: Center of the tube lid
                    kp1_x = row['center_x'] / img_w
                    kp1_y = row['center_y'] / img_h
                    
                    # Keypoint 2: The Tab
                    # Project outward from center using the angle. (Y is subtracted because Y goes down in images)
                    angle_rad = math.radians(row['angle_deg'])
                    radius = max(row['bbox_w'], row['bbox_h']) / 2.0 
                    kp2_x = (row['center_x'] + radius * math.cos(angle_rad)) / img_w
                    kp2_y = (row['center_y'] - radius * math.sin(angle_rad)) / img_h
                    
                    # Ensure keypoints are within 0-1 bounds
                    kp1_x, kp1_y = max(0, min(1, kp1_x)), max(0, min(1, kp1_y))
                    kp2_x, kp2_y = max(0, min(1, kp2_x)), max(0, min(1, kp2_y))
                    
                    # Format: class bbox_cx bbox_cy w h kp1_x kp1_y vis1 kp2_x kp2_y vis2
                    # Visibility flag '2' means visible and labeled
                    line = f"0 {bbox_cx:.5f} {bbox_cy:.5f} {norm_w:.5f} {norm_h:.5f} {kp1_x:.5f} {kp1_y:.5f} 2 {kp2_x:.5f} {kp2_y:.5f} 2\n"
                    f.write(line)

    process_split(train_imgs, 'train')
    process_split(val_imgs, 'val')

def create_yaml():
    yaml_path = 'dataset_pose.yaml'
    abs_dir = os.path.abspath(YOLO_DATASET_DIR) 
    
    # new kpt_shape parameter: [2 keypoints, 3 dimensions (x, y, visibility)]
    yaml_content = f"""
path: {abs_dir}
train: images/train
val: images/val

names:
  0: tube

kpt_shape: [2, 3] 
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    return yaml_path

if __name__ == "__main__":
    setup_yolo_directories()
    convert_and_split_data(test_size=10/70)
    yaml_path = create_yaml()
    
    # Load the POSE model
    model = YOLO('yolov8n-pose.pt')
    
    results = model.train(
        data=yaml_path,
        epochs=150,
        imgsz=640,
        fliplr=0.0,
        flipud=0.0,
        degrees=90.0,
        batch=8
    )