### 1. Installation
Ensure you have the required libraries installed:
```bash
pip install ultralytics pandas numpy scipy opencv-python scikit-learn
```
### 2. Repo structure
*   `train.py` - Handles dataset splitting, YOLO format conversion, and triggers the fine-tuning process using the Ultralytics library.
*   `evaluate.py` - Runs inference on test images, performs bipartite matching for evaluation, calculates custom error metrics, and outputs visual OpenCV representations to `output/`.
*   `data/` - Directory for raw images and annotations.
*   `predict.py` - takes input images from `pred_in/`, predicts the tubes and sends output to `pred_out/`.
