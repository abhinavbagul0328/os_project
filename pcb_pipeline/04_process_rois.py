import os
import cv2
import numpy as np
from utils import ensure_dirs, ROI_DATASET_DIR, ROI_PROCESSED_DIR, load_image, load_json, save_json

def compute_cv_features(img_path, save_dir, basename):
    """
    Computes CV features (Canny edges, ORB descriptors, intensity stats)
    and saves intermediate debug images if save_dir is provided.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
        
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    features = {}
    
    # 1. Intensity Stats
    features["mean_intensity"] = float(np.mean(img_gray))
    features["std_intensity"] = float(np.std(img_gray))
    features["min_intensity"] = float(np.min(img_gray))
    features["max_intensity"] = float(np.max(img_gray))
    
    # 1.5 Color Average Pixel Values
    features["mean_B"] = float(np.mean(img_bgr[:,:,0]))
    features["mean_G"] = float(np.mean(img_bgr[:,:,1]))
    features["mean_R"] = float(np.mean(img_bgr[:,:,2]))
    features["average_pixel_value_rgb"] = float((features["mean_R"] + features["mean_G"] + features["mean_B"]) / 3.0)
    
    # 2. Canny Edge Detection
    # Use Otsu's method to find optimal thresholds for Canny
    high_thresh, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(img_gray, low_thresh, high_thresh)
    
    edge_density = np.sum(edges > 0) / edges.size
    features["canny_edge_density"] = float(edge_density)
    
    # 3. ORB (Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    
    features["orb_keypoint_count"] = len(keypoints)
    
    if descriptors is not None and len(descriptors) > 0:
        # Descriptors are uint8, compute mean/std across all descriptors
        features["orb_descriptor_mean"] = float(np.mean(descriptors))
        features["orb_descriptor_std"] = float(np.std(descriptors))
    else:
        features["orb_descriptor_mean"] = 0.0
        features["orb_descriptor_std"] = 0.0
        
    # Save visual outputs if requested
    if save_dir and basename:
        cv2.imwrite(os.path.join(save_dir, f"{basename}_canny.png"), edges)
        
        # Draw keypoints
        img_kp = cv2.drawKeypoints(img_bgr, keypoints, None, color=(0, 255, 0), flags=0)
        cv2.imwrite(os.path.join(save_dir, f"{basename}_orb.png"), img_kp)
        
    return features

def main():
    ensure_dirs()
    
    roi_meta_path = os.path.join(ROI_DATASET_DIR, "roi_metadata.json")
    roi_metadata = load_json(roi_meta_path)
    
    if not roi_metadata:
        print("Error: roi_metadata.json not found. Did you run 03_extract_rois.py?")
        return
        
    print(f"Processing {len(roi_metadata)} ROIs to extract CV features...")
    
    all_roi_features = []
    
    for i, item in enumerate(roi_metadata):
        roi_filepath = item["roi_filepath"]
        orig_img = item["original_image"]
        roi_id = item["roi_id"]
        
        # Output paths
        img_basename = os.path.splitext(orig_img)[0]
        out_save_dir = os.path.join(ROI_PROCESSED_DIR, img_basename)
        os.makedirs(out_save_dir, exist_ok=True)
        
        roi_basename = f"roi_{roi_id:02d}"
        
        # Compute features
        features = compute_cv_features(roi_filepath, out_save_dir, roi_basename)
        
        if features:
            # Combine original metadata with new features
            combined = item.copy()
            combined.update(features)
            combined["processed_dir"] = out_save_dir
            all_roi_features.append(combined)
            
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(roi_metadata)} ROIs...")
            
    features_meta_path = os.path.join(ROI_PROCESSED_DIR, "roi_features.json")
    save_json(features_meta_path, all_roi_features)
    print(f"Finished processing. Features saved to {features_meta_path}")

if __name__ == "__main__":
    main()
