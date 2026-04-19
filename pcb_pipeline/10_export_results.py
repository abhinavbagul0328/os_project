import os
import importlib.util
from utils import ensure_dirs, OUTPUT_DIR, load_json

# Dynamic import for numbered file
spec = importlib.util.spec_from_file_location("step07", os.path.join(os.path.dirname(__file__), "07_predict.py"))
step07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step07)
predict_image = step07.predict_image
load_all_roi_models = step07.load_all_roi_models

def main():
    print("--- STARTING FINAL METADATA EXPORT ---")
    
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        print("Error: metadata.json not found.")
        return
        
    metadata = load_json(metadata_path)
    
    export_dir = os.path.join(OUTPUT_DIR, "export")
    os.makedirs(export_dir, exist_ok=True)
    
    # Load models once to speed up predictions
    roi_models = load_all_roi_models()
    if not roi_models:
        print("Error: No models found to export results with.")
        return
        
    total = len(metadata)
    print(f"Exporting precise metadata reports for {total} images...")
    
    for idx, (img_name, info) in enumerate(metadata.items(), 1):
        img_path = info["path"]
        
        # Run the massive prediction pipeline on it silently
        img_rgb, img_is_faulty, roi_results, _ = predict_image(img_path, roi_models=roi_models)
        
        if img_rgb is None:
            continue
            
        txt_path = os.path.join(export_dir, f"{os.path.splitext(img_name)[0]}_report.txt")
        
        with open(txt_path, "w") as f:
            f.write("="*40 + "\n")
            f.write(f"PCB ANALYSIS REPORT: {img_name}\n")
            f.write("="*40 + "\n\n")
            
            f.write("--- IMAGE METADATA ---\n")
            f.write(f"Original Label: {info.get('original_label', 'UNKNOWN')}\n")
            f.write(f"File Path: {img_path}\n")
            f.write(f"Predicted Status: {'FAULTY' if img_is_faulty else 'CLEAN'}\n\n")
            
            f.write("--- ROI BREAKDOWN ---\n")
            for res in roi_results:
                x1, y1, x2, y2 = res["coords"]
                status = "FAULTY" if res["faulty"] else "CLEAN"
                f.write(f"ROI ID: {res['id']:02d}\n")
                f.write(f"  Coordinates: {x1},{y1} to {x2},{y2}\n")
                f.write(f"  Status:      {status}\n")
                f.write("-" * 20 + "\n")
                
        if idx % 10 == 0 or idx == total:
            print(f"  Exported {idx}/{total} files...")
            
    print(f"\n[SUCCESS] All metadata texts exported successfully to: {export_dir}")

if __name__ == "__main__":
    main()
