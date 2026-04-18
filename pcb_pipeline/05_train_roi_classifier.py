import os
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from utils import ensure_dirs, ROI_PROCESSED_DIR, MODELS_DIR, load_json

def prepare_dataset_by_roi(features_data):
    # Maps roi_id -> { "X": [], "roi_info": [] }
    roi_groups = defaultdict(lambda: {"X": [], "roi_info": []})
    
    feature_keys = [
        "mean_intensity",
        "std_intensity",
        "min_intensity",
        "max_intensity",
        "mean_R",
        "mean_G",
        "mean_B",
        "average_pixel_value_rgb",
        "canny_edge_density", 
        "orb_keypoint_count", 
        "orb_descriptor_mean", 
        "orb_descriptor_std"
    ]
    
    for item in features_data:
        roi_id = item["roi_id"]
        # Build feature vector
        vector = [item.get(k, 0.0) for k in feature_keys]
        
        roi_groups[roi_id]["X"].append(vector)
        roi_groups[roi_id]["roi_info"].append({
            "image": item["original_image"],
            "roi_id": roi_id
        })
        
    # Convert lists to NumPy arrays
    for roi_id in roi_groups:
        roi_groups[roi_id]["X"] = np.array(roi_groups[roi_id]["X"])
        
    return roi_groups, feature_keys

def main():
    ensure_dirs()
    
    features_path = os.path.join(ROI_PROCESSED_DIR, "roi_features.json")
    features_data = load_json(features_path)
    
    if not features_data:
        print("Error: roi_features.json not found. Did you run 04_process_rois.py?")
        return
        
    roi_groups, feature_names = prepare_dataset_by_roi(features_data)
    print(f"Dataset prepared: found {len(roi_groups)} distinct ROI locations.")
    
    for roi_id, data in roi_groups.items():
        X = data["X"]
        print(f"\n--- Training Unsupervised KMeans for ROI {roi_id} ---")
        print(f"Data shape: {X.shape[0]} ROIs, {X.shape[1]} features")
        
        # Scale features for KMeans
        scaler = StandardScaler()
        if X.shape[0] < 2:
            print(f"Skipping ROI {roi_id}: mathematically impossible to cluster ({X.shape[0]} samples).")
            continue
            
        X_scaled = scaler.fit_transform(X)
        
        # Give 10x weight to average_pixel_value_rgb
        avg_pixel_idx = feature_names.index("average_pixel_value_rgb")
        X_scaled[:, avg_pixel_idx] *= 10000.0
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Determine which cluster is the minority
        count_0 = np.sum(cluster_labels == 0)
        count_1 = np.sum(cluster_labels == 1)
        
        if count_0 < count_1:
            faulty_cluster = 0
            clean_cluster = 1
        else:
            faulty_cluster = 1
            clean_cluster = 0
            
        print(f"Clustering Results for ROI {roi_id}:")
        print(f"  Cluster {clean_cluster} (Majority/Clean): {np.sum(cluster_labels == clean_cluster)} ROIs")
        print(f"  Cluster {faulty_cluster} (Minority/Faulty): {np.sum(cluster_labels == faulty_cluster)} ROIs")
        
        # Save Model locally for this specific ROI
        model_data = {
            'roi_id': roi_id,
            'model': kmeans,
            'scaler': scaler,
            'feature_names': feature_names,
            'faulty_cluster': faulty_cluster,
            'clean_cluster': clean_cluster
        }
        
        model_path = os.path.join(MODELS_DIR, f"roi_classifier_{roi_id}.pkl")
        joblib.dump(model_data, model_path)
        print(f"Saved ROI {roi_id} model to {model_path}")
        
    print("\nFinished training all ROI-specific models.")

if __name__ == "__main__":
    main()
