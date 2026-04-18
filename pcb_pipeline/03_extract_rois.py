import os
import cv2
from utils import ensure_dirs, OUTPUT_DIR, ROI_DATASET_DIR, load_image, load_json, save_json

def crop_roi(image, x1, y1, x2, y2):
    """Safely crop ROI from image handling bounds."""
    h, w = image.shape[:2]
    # Bound coordinates
    x1, x2 = sorted([max(0, min(x1, w)), max(0, min(x2, w))])
    y1, y2 = sorted([max(0, min(y1, h)), max(0, min(y2, h))])
    
    if x2 <= x1 or y2 <= y1:
        return None # Invalid crop
    
    return image[y1:y2, x1:x2]

def main():
    ensure_dirs()
    
    metadata = load_json(os.path.join(OUTPUT_DIR, "metadata.json"))
    roi_data = load_json(os.path.join(OUTPUT_DIR, "roi_coordinates.json"))
    
    if not metadata:
        print("Error: metadata.json not found.")
        return
        
    if not roi_data or "rois" not in roi_data or len(roi_data["rois"]) == 0:
        print("Error: valid roi_coordinates.json not found. Did you run 02_select_rois.py?")
        return
        
    rois = roi_data["rois"]
    print(f"Loaded {len(rois)} ROI definitions. Extracting from {len(metadata)} images...")
    
    all_roi_metadata = []
    
    # Process each image
    for img_filename, img_data in metadata.items():
        img_path = img_data['path']
        img_label = img_data['binary_label']
        
        # Load image once per file
        img = load_image(img_path)
        if img is None:
            print(f"Skipping {img_filename}, failed to load.")
            continue
            
        # Create output dir for this image's ROIs
        img_roi_dir = os.path.join(ROI_DATASET_DIR, os.path.splitext(img_filename)[0])
        os.makedirs(img_roi_dir, exist_ok=True)
        
        for roi_def in rois:
            roi_id = roi_def["roi_id"]
            x1, y1 = roi_def["x1"], roi_def["y1"]
            x2, y2 = roi_def["x2"], roi_def["y2"]
            
            cropped = crop_roi(img, x1, y1, x2, y2)
            if cropped is None:
                print(f"Warning: Invalid crop for ROI {roi_id} in {img_filename}")
                continue
                
            roi_filename = f"roi_{roi_id:02d}.png"
            roi_filepath = os.path.join(img_roi_dir, roi_filename)
            
            # Save cropped image (convert back to BGR for cv2)
            cv2.imwrite(roi_filepath, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            
            # Record metadata
            all_roi_metadata.append({
                "roi_filepath": roi_filepath,
                "original_image": img_filename,
                "label": img_label, # Use same label as parent image
                "defect_type": img_data['label'],
                "roi_id": roi_id,
                "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })
            
    roi_meta_path = os.path.join(ROI_DATASET_DIR, "roi_metadata.json")
    save_json(roi_meta_path, all_roi_metadata)
    print(f"Extracted {len(all_roi_metadata)} ROIs. Metadata saved to {roi_meta_path}")

if __name__ == "__main__":
    main()
