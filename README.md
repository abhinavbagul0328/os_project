# PCB Slider Anomaly Detection Pipeline

This directory contains a complete pipeline to ingest images of Printed Circuit Boards (PCBs), automatically generate anomaly detection models targeting specific regions of interest (ROIs), and classify full images as `Clean (OK)` or `Faulty (NG)` using localized algorithms.

## Workflow Execution

The pipeline is entirely modular and should be run in sequence from within the `pcb_pipeline` directory.

### 1. Data Inspection
`python 01_load_and_inspect.py`
Scans the `data_pro/` directory, tallies up the dataset, and builds baseline metadata.

### 2. Interactive ROI Selection
`python 02_select_rois.py`
Launches an interactive GUI built on `matplotlib` to allow the user to draw boxes over critical areas of the PCB. These coordinate boundaries are saved globally across the pipeline.

### 3. ROI Extraction
`python 03_extract_rois.py`
Reads the selected ROI coordinates and automatically crops out those distinct sub-images for all samples in the dataset.

### 4. Feature Processing
`python 04_process_rois.py`
Calculates highly specific Computer Vision metrics for every ROI, such as:
- Standard intensity
- Color (BGR) averages
- Canny edge density
- ORB keypoint counts and descriptor stats

### 5. Independent ROI Model Training
`python 05_train_roi_classifier.py`
Ingests all feature metrics and clusters them sequentially. Instead of training one model, this script dynamically trains **one model per ROI location**, employing Unsupervised KMeans logic to split datasets into Clean vs Faulty subsets independent of label strings.

### 6. Full Set Image Classification
`python 06_classify_images.py`
Evaluates the entire dataset architecture using the distinct dynamic models for each corner/position, printing out total validation accuracy data.

### 7. Single Image Prediction
`python 07_predict.py --image ../data_pro/1.png --visualize`
Takes a single target image, extracts its independent ROIs, routes those localized image data chunks to their proper models, and draws color-coded rectangles denoting safe or anomalous regions directly over the image.
