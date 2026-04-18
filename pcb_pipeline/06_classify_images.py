import os
import joblib
from utils import ensure_dirs, OUTPUT_DIR, MODELS_DIR, load_json, save_json

def load_all_roi_models():
    """Loads all roi_classifier_X.pkl models from the models directory."""
    models = {}
    if not os.path.exists(MODELS_DIR):
        return models
        
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith("roi_classifier_") and filename.endswith(".pkl"):
            # Extract ROI ID from filename (e.g. roi_classifier_2.pkl -> 2)
            try:
                roi_id_str = filename.split("_")[-1].split(".")[0]
                roi_id = int(roi_id_str)
                model_path = os.path.join(MODELS_DIR, filename)
                models[roi_id] = joblib.load(model_path)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                
    return models

def main():
    ensure_dirs()
    
    features_data = load_json(os.path.join(OUTPUT_DIR, "roi_processed", "roi_features.json"))
    
    if not features_data:
        print("Error: Missing features data.")
        return
        
    roi_models = load_all_roi_models()
    if not roi_models:
        print("Error: No ROI models found. Did you run 05_train_roi_classifier.py?")
        return
        
    print(f"Loaded {len(roi_models)} ROI-specific clustering models.")
    
    # Group ROI features by image
    image_roi_map = {}
    for item in features_data:
        img_name = item["original_image"]
        if img_name not in image_roi_map:
            image_roi_map[img_name] = []
        image_roi_map[img_name].append(item)
        
    print(f"Classifying {len(image_roi_map)} images using individual ROI models...\n")
    
    results = {}
    faulty_images_count = 0
    clean_images_count = 0
    
    for img_name, rois in image_roi_map.items():
        image_is_faulty = False
        roi_predictions = []
        
        for roi in rois:
            roi_id = roi["roi_id"]
            
            if roi_id not in roi_models:
                print(f"Warning: No model found for ROI {roi_id} in image {img_name}. Assuming CLEAN.")
                roi_predictions.append({
                    "roi_id": roi_id,
                    "prediction": "clean",
                    "note": "missing model"
                })
                continue
                
            model_data = roi_models[roi_id]
            kmeans = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            faulty_cluster = model_data['faulty_cluster']
            
            vector = [[roi.get(k, 0.0) for k in feature_names]]
            vector_scaled = scaler.transform(vector)
            
            # Give 10x weight to average_pixel_value_rgb
            avg_pixel_idx = feature_names.index("average_pixel_value_rgb")
            vector_scaled[0, avg_pixel_idx] *= 10000.0
            
            pred_cluster = kmeans.predict(vector_scaled)[0]
            
            is_faulty = (pred_cluster == faulty_cluster)
            roi_predictions.append({
                "roi_id": roi_id,
                "prediction": "faulty" if is_faulty else "clean"
            })
            
            if is_faulty:
                image_is_faulty = True
                
        if image_is_faulty:
            faulty_images_count += 1
        else:
            clean_images_count += 1
            
        results[img_name] = {
            "predicted_label": "faulty" if image_is_faulty else "clean",
            "roi_breakdown": roi_predictions
        }
        
    print(f"--- Full Dataset Priority ROI Classification Results ---")
    print(f"Total Images Analyzed : {len(image_roi_map)}")
    print(f"Images marked CLEAN   : {clean_images_count}")
    print(f"Images marked FAULTY  : {faulty_images_count}")
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, "image_classification_results.json")
    save_json(results_path, {
        "summary": {
            "total_images": len(image_roi_map),
            "clean_images": clean_images_count,
            "faulty_images": faulty_images_count
        },
        "details": results
    })
    print(f"\nDetailed image-level results saved to {results_path}")

if __name__ == "__main__":
    main()
