import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, accuracy_score
import importlib.util
import copy

# Helper to import files that start with numbers
def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_dir = os.path.dirname(os.path.abspath(__file__))
utils = import_script("utils", os.path.join(base_dir, "utils.py"))
step07 = import_script("step07", os.path.join(base_dir, "07_predict.py"))

class EvalInteractiveHandler:
    def __init__(self, fig, ax, roi_results):
        self.fig = fig
        self.ax = ax
        self.roi_results = roi_results # list of dicts: {"id", "coords", "faulty"}
        self.rect_patches = []
        self.text_patches = []
        self.draw()
        self.cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

    def draw(self):
        [p.remove() for p in self.rect_patches]
        [t.remove() for t in self.text_patches]
        self.rect_patches.clear()
        self.text_patches.clear()

        for res in self.roi_results:
            color = '#D9534F' if res["faulty"] else '#5CB85C'
            x1, y1, x2, y2 = res["coords"]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor=color, facecolor='none')
            self.ax.add_patch(rect)
            self.rect_patches.append(rect)
            txt = self.ax.text(x1, y1 - 10, f"ROI {res['id']}: {'NG' if res['faulty'] else 'OK'}", 
                               color=color, fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.7))
            self.text_patches.append(txt)
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        cx, cy = event.xdata, event.ydata
        if cx is None or cy is None: return

        for res in self.roi_results:
            x1, y1, x2, y2 = res["coords"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                res["faulty"] = not res["faulty"]
                self.draw()
                break

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['CLEAN ROI', 'FAULTY ROI']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'ROI-Level Performance Matrix\nFinal Accuracy: {acc:.2%}',
           ylabel='Human Truth (Your Correction)',
           xlabel='AI Prediction (Initial Guess)')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()

def main():
    print("="*50)
    print("   PCB ROI-LEVEL EVALUATION TOOL (STEP 11)   ")
    print("="*50)
    
    metadata_path = os.path.join(utils.OUTPUT_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        print("Error: metadata.json not found. Please run Step 1 first.")
        return
    
    metadata = utils.load_json(metadata_path)
    all_image_paths = [info["path"] for info in metadata.values()]
    
    try:
        num_eval = int(input("How many random images to audit for ROI Accuracy? (default 5): ") or 5)
    except ValueError:
        num_eval = 5
        
    eval_subset = random.sample(all_image_paths, min(num_eval, len(all_image_paths)))
    
    y_true_rois = []
    y_pred_rois = []
    
    print("\nStarting ROI Evaluation Session...")
    print("Instructions: Correct any wrong ROI labels by clicking them. Close window to save results.")
    
    for i, img_path in enumerate(eval_subset, 1):
        # 1. Get Initial AI Prediction
        img_rgb, _, roi_results, _ = step07.predict_image(img_path)
        
        if img_rgb is None:
            continue
            
        # Capture exactly what the AI predicted for each ROI before human intervention
        ai_initial_roi_guesses = [1 if r["faulty"] else 0 for r in roi_results]
        
        # 2. Open Window for Human Verification
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(img_rgb)
        ax.set_title(f"Image [{i}/{num_eval}]: {os.path.basename(img_path)}\nFlip WRONG ROI labels by clicking. Close window to confirm.")
        
        handler = EvalInteractiveHandler(fig, ax, roi_results)
        plt.show()
        
        # 3. Capture Human Truth
        human_truth_rois = [1 if r["faulty"] else 0 for r in roi_results]
        
        # Add to global lists
        y_pred_rois.extend(ai_initial_roi_guesses)
        y_true_rois.extend(human_truth_rois)
        
        print(f"  Processed {len(roi_results)} ROIs on image {i}.")
        
    if not y_true_rois:
        print("\nNo data collected.")
        return
        
    print("\n" + "="*40)
    print("FINAL ROI-LEVEL ANALYTICS")
    print("="*40)
    print(f"Total ROIs Audited:    {len(y_true_rois)}")
    print(f"Overall ROI Accuracy: {accuracy_score(y_true_rois, y_pred_rois):.2%}")
    
    plot_confusion_matrix(y_true_rois, y_pred_rois)

if __name__ == "__main__":
    main()
