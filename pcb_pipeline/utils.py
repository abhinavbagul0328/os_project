import os
import cv2
import json

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_pro'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
ROI_DATASET_DIR = os.path.join(OUTPUT_DIR, 'roi_dataset')
ROI_PROCESSED_DIR = os.path.join(OUTPUT_DIR, 'roi_processed')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

_mapping_cache = None

def get_mapping():
    global _mapping_cache
    if _mapping_cache is not None:
        return _mapping_cache
    
    mapping_path = os.path.join(DATA_DIR, "name_mapping.csv")
    _mapping_cache = {}
    if os.path.exists(mapping_path):
        import csv
        with open(mapping_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                _mapping_cache[row['NewName']] = row['OriginalName']
    return _mapping_cache

def parse_label_from_filename(filename):
    """
    Parses the condition by looking up the original filename in name_mapping.csv
    and extracting the label based on the pattern: <date>-pcb-Pcb_slider-<OK|NGxx>.png
    """
    mapping = get_mapping()
    original_name = mapping.get(filename, filename)
    
    basename = os.path.basename(original_name)
    name_no_ext = os.path.splitext(basename)[0]
    parts = name_no_ext.split('-')
    label_part = parts[-1] if len(parts) > 0 else "UNKNOWN"
    
    if "OK" in label_part:
        return "OK"
    elif "NG" in label_part:
        return label_part # return specific defect class like NG02
    else:
        return "UNKNOWN"

def get_binary_label(label_part):
    return "clean" if label_part == "OK" else "faulty"

def load_image(filepath):
    """
    Load an image from disk in RGB format.
    """
    img = cv2.imread(filepath)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ROI_DATASET_DIR, exist_ok=True)
    os.makedirs(ROI_PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)
