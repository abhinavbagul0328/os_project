import os
import random
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
from utils import ensure_dirs, OUTPUT_DIR, load_image, save_json, load_json

class ROISelector:
    def __init__(self, ax, image_name):
        self.ax = ax
        self.image_name = image_name
        self.rois = [] # List of tuples: (x1, y1, x2, y2)
        self.rect_patches = []
        self.text_patches = []
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']
        
        self.RS = RectangleSelector(
            ax, self.line_select_callback,
            useblit=True,
            button=[1], # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

    def line_select_callback(self, eclick, erelease):
        """Callback for line selection."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Ensure x1 < x2 and y1 < y2
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        roi_id = len(self.rois) + 1
        color = self.colors[roi_id % len(self.colors)]
        
        self.rois.append((x_min, y_min, x_max, y_max))
        
        # Draw rectangle
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                 linewidth=2, edgecolor=color, facecolor='none')
        self.ax.add_patch(rect)
        self.rect_patches.append(rect)
        
        # Add text
        text = self.ax.text(x_min, y_min - 10, f"ROI {roi_id}", color=color, 
                            fontsize=12, weight='bold', bbox=dict(facecolor='white', alpha=0.5))
        self.text_patches.append(text)
        
        self.ax.figure.canvas.draw()
        print(f"Added ROI {roi_id}: x1={x_min}, y1={y_min}, x2={x_max}, y2={y_max}")

def main():
    ensure_dirs()
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    metadata = load_json(metadata_path)
    
    if not metadata:
        print("Error: metadata.json not found. Please run 01_load_and_inspect.py first.")
        return
        
    # Get all images
    image_names = list(metadata.keys())
    
    # Pick a random image (prefer OK for template definition if possible)
    ok_images = [name for name, data in metadata.items() if data['binary_label'] == "clean"]
    if ok_images:
        target_image_name = random.choice(ok_images)
    else:
        target_image_name = random.choice(image_names)
        
    target_image_path = metadata[target_image_name]['path']
    print(f"Loading image for ROI selection: {target_image_name}")
    
    img = load_image(target_image_path)
    if img is None:
        print(f"Failed to load image: {target_image_path}")
        return
        
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(f"ROI Selection: {target_image_name}\nClick and drag to select regions. Press 'Q' or close window to save.")
    
    selector = ROISelector(ax, target_image_name)
    
    plt.show() # Blocks until window is closed
    
    print("\nSelection finished.")
    if selector.rois:
        output_data = {
            "demo_image": target_image_name,
            "rois": []
        }
        for idx, (x1, y1, x2, y2) in enumerate(selector.rois):
            output_data["rois"].append({
                "roi_id": idx + 1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })
            
        roi_output_path = os.path.join(OUTPUT_DIR, "roi_coordinates.json")
        save_json(roi_output_path, output_data)
        print(f"Saved {len(selector.rois)} ROIs to {roi_output_path}")
    else:
        print("No ROIs selected. File not saved.")

if __name__ == "__main__":
    main()
