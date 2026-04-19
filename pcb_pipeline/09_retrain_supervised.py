import os
import cv2
import numpy as np
import joblib
import importlib.util
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from utils import ensure_dirs, OUTPUT_DIR, MODELS_DIR, load_json, load_image

# Dynamic import for numbered file
spec = importlib.util.spec_from_file_location("step07", os.path.join(os.path.dirname(__file__), "07_predict.py"))
step07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step07)
crop_roi = step07.crop_roi
extract_features = step07.extract_features
load_all_roi_models = step07.load_all_roi_models

def main():
    print("--- STARTING SUPERVISED RETRAINING WITH PSEUDO-LABELING ---")
    feedback_path = os.path.join(OUTPUT_DIR, "feedback_labels.json")
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    roi_coords_path = os.path.join(OUTPUT_DIR, "roi_coordinates.json")
    
    if not os.path.exists(feedback_path):
        print("Error: No feedback_labels.json found. Please do Manual Training in Step 8 first.")
        return
        
    feedback_data = load_json(feedback_path)
    metadata = load_json(metadata_path)
    roi_coords_data = load_json(roi_coords_path)
    
    if not feedback_data:
        print("Error: feedback_labels.json is empty.")
        return
        
    old_models = load_all_roi_models()
    
    print(f"Loaded Explicit Truth labels for {len(feedback_data)} images.")
    print(f"Generating Pseudo-labels for remaining {len(metadata) - len(feedback_data)} images globally...")
    
    roi_datasets = {}
    roi_box_map = {r["roi_id"]: (r["x1"], r["y1"], r["x2"], r["y2"]) for r in roi_coords_data["rois"]}
    
    for count, (img_name, info) in enumerate(metadata.items(), 1):
        img_path = info["path"]
        img_rgb = load_image(img_path)
        if img_rgb is None:
            continue
            
        explicit_rois = {}
        if img_path in feedback_data:
            explicit_rois = {r["id"]: r["faulty"] for r in feedback_data[img_path]}
            
        for r_id, coords in roi_box_map.items():
            x1, y1, x2, y2 = coords
            cropped_rgb = crop_roi(img_rgb, x1, y1, x2, y2)
            if cropped_rgb is None:
                continue
                
            cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
            feats_dict = extract_features(cropped_bgr)
            
            if r_id not in roi_datasets:
                roi_datasets[r_id] = {"X": [], "y": [], "features": list(feats_dict.keys())}
                
            feature_vector = [feats_dict[k] for k in roi_datasets[r_id]["features"]]
            
            label = 0
            if r_id in explicit_rois:
                # 1. Use User's Explicit Ground Truth
                label = 1 if explicit_rois[r_id] else 0
            else:
                # 2. Use Old Unsupervised Model's Prediction
                if r_id in old_models:
                    model_data = old_models[r_id]
                    scaler = model_data['scaler']
                    kmeans = model_data['model']
                    fc = model_data['faulty_cluster']
                    
                    vector_scaled = scaler.transform([feature_vector])
                    avg_idx = roi_datasets[r_id]["features"].index("average_pixel_value_rgb")
                    vector_scaled[0, avg_idx] *= 1000.0
                    
                    pred_cluster = kmeans.predict(vector_scaled)[0]
                    label = 1 if pred_cluster == fc else 0
                    
            roi_datasets[r_id]["X"].append(feature_vector)
            roi_datasets[r_id]["y"].append(label)

    print("\nExtracted truth bounds & pseudo labels successfully! Injecting logic into Supervised Models...")
    
    ensure_dirs()
    
    for r_id, data in roi_datasets.items():
        X = np.array(data["X"])
        y = np.array(data["y"])
        feature_names = data["features"]
        
        if len(np.unique(y)) < 2:
            print(f"  WARNING: ROI {r_id} only has examples of one class across BOTH user truth and old pseudo data! Math boundary impossible.")
            continue
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        avg_pixel_idx = feature_names.index("average_pixel_value_rgb")
        X_scaled[:, avg_pixel_idx] *= 1000.0
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf.fit(X_scaled, y)
        
        packaged_model = {
            'model': rf,
            'scaler': scaler,
            'feature_names': feature_names,
            'faulty_cluster': 1
        }
        
        model_path = os.path.join(MODELS_DIR, f"roi_classifier_{r_id}.pkl")
        joblib.dump(packaged_model, model_path)
        print(f"  [SUCCESS] ROI {r_id}: Trained highly robust Random Forest explicitly across {len(X)} labeled examples!")

    print("\n--- SUPERVISED RETRAINING COMPLETE ---")

if __name__ == "__main__":
    main()
