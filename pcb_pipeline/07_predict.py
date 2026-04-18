import os
import argparse
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import ensure_dirs, OUTPUT_DIR, MODELS_DIR, load_image, load_json

def crop_roi(image, x1, y1, x2, y2):
    h, w = image.shape[:2]
    x1, x2 = sorted([max(0, min(x1, w)), max(0, min(x2, w))])
    y1, y2 = sorted([max(0, min(y1, h)), max(0, min(y2, h))])
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]

def extract_features(roi_bgr):
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    features = {}
    features["mean_intensity"] = float(np.mean(roi_gray))
    features["std_intensity"] = float(np.std(roi_gray))
    features["min_intensity"] = float(np.min(roi_gray))
    features["max_intensity"] = float(np.max(roi_gray))
    
    features["mean_B"] = float(np.mean(roi_bgr[:,:,0]))
    features["mean_G"] = float(np.mean(roi_bgr[:,:,1]))
    features["mean_R"] = float(np.mean(roi_bgr[:,:,2]))
    features["average_pixel_value_rgb"] = float((features["mean_R"] + features["mean_G"] + features["mean_B"]) / 3.0)
    
    high_thresh, _ = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(roi_gray, low_thresh, high_thresh)
    features["canny_edge_density"] = float(np.sum(edges > 0) / edges.size)
    
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(roi_gray, None)
    features["orb_keypoint_count"] = len(keypoints)
    if descriptors is not None and len(descriptors) > 0:
        features["orb_descriptor_mean"] = float(np.mean(descriptors))
        features["orb_descriptor_std"] = float(np.std(descriptors))
    else:
        features["orb_descriptor_mean"] = 0.0
        features["orb_descriptor_std"] = 0.0
        
    return features

def load_all_roi_models():
    """Loads all roi_classifier_X.pkl models."""
    models = {}
    if not os.path.exists(MODELS_DIR):
        return models
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith("roi_classifier_") and filename.endswith(".pkl"):
            try:
                roi_id_str = filename.split("_")[-1].split(".")[0]
                roi_id = int(roi_id_str)
                model_path = os.path.join(MODELS_DIR, filename)
                models[roi_id] = joblib.load(model_path)
            except Exception:
                pass
    return models

def main():
    parser = argparse.ArgumentParser(description="Predict if a PCB image is clean or faulty based on distinct ROI models.")
    parser.add_argument("--image", required=True, help="Path to the image to analyze")
    parser.add_argument("--visualize", action="store_true", help="Show an image with overlayed ROIs")
    args = parser.parse_args()
    
    img_path = args.image
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
        
    roi_data_path = os.path.join(OUTPUT_DIR, "roi_coordinates.json")
    if not os.path.exists(roi_data_path):
        print("Error: Missing ROI coordinates.")
        return
        
    roi_models = load_all_roi_models()
    if not roi_models:
        print("Error: No localized ROI models found. Run step 5 first.")
        return
        
    # Load ROIs
    roi_data = load_json(roi_data_path)
    rois = roi_data["rois"]
    
    # Load Image
    img_rgb = load_image(img_path)
    if img_rgb is None:
        print("Failed to load image.")
        return
        
    print(f"Analyzing {os.path.basename(img_path)} against {len(rois)} distinct ROI models...")
    
    img_is_faulty = False
    roi_results = []
    
    for roi in rois:
        r_id = roi["roi_id"]
        x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
        
        cropped_rgb = crop_roi(img_rgb, x1, y1, x2, y2)
        if cropped_rgb is None:
            print(f"ROI {r_id} out of bounds.")
            continue
            
        cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
        feats_dict = extract_features(cropped_bgr)
        
        # Check if we have a model for this specific ROI
        if r_id not in roi_models:
            print(f"  ROI {r_id:02d}: SKIPPED (No dedicated model found)")
            continue
            
        model_data = roi_models[r_id]
        kmeans = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        faulty_cluster = model_data['faulty_cluster']
        
        vector = [[feats_dict.get(k, 0.0) for k in feature_names]]
        vector_scaled = scaler.transform(vector)
        
        # Give 10x weight to average_pixel_value_rgb
        avg_pixel_idx = feature_names.index("average_pixel_value_rgb")
        vector_scaled[0, avg_pixel_idx] *= 10000.0
        
        pred_cluster = kmeans.predict(vector_scaled)[0]
        
        is_faulty = (pred_cluster == faulty_cluster)
        if is_faulty:
            img_is_faulty = True
            
        roi_results.append({
            "id": r_id, "coords": (x1, y1, x2, y2), "faulty": is_faulty
        })
        
        status = "FAULTY" if is_faulty else "CLEAN"
        print(f"  ROI {r_id:02d}: {status}")

    print("\n" + "="*30)
    print(f"FINAL DECISION: {'FAULTY (NG)' if img_is_faulty else 'CLEAN (OK)'}")
    print("="*30)

    if args.visualize:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_rgb)
        ax.set_title(f"Prediction: {'FAULTY (NG)' if img_is_faulty else 'CLEAN (OK)'}")
        
        for res in roi_results:
            color = 'red' if res["faulty"] else 'green'
            x1, y1, x2, y2 = res["coords"]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                     linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 10, f"ROI {res['id']}: {'NG' if res['faulty'] else 'OK'}", 
                    color=color, fontsize=12, weight='bold', bbox=dict(facecolor='white', alpha=0.7))
        plt.show()

if __name__ == "__main__":
    main()
