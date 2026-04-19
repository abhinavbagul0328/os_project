import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FeedbackSelector:
    """
    Handles interactive click events to flip ROI status between Faulty/NG and Clean/OK.
    """
    def __init__(self, canvas, ax, image_name, roi_results):
        self.canvas = canvas
        self.ax = ax
        self.image_name = image_name
        self.roi_results = roi_results  # list of dict: {"id": r_id, "coords": (x1, y1, x2, y2), "faulty": is_faulty}
        self.rect_patches = []
        self.text_patches = []
        
        self.draw_rois()
        self.cid = self.canvas.mpl_connect('button_press_event', self.on_click)
        
    def draw_rois(self):
        # Clear specific patches inside the ax to redraw freshly without wiping the image
        [p.remove() for p in self.rect_patches]
        [t.remove() for t in self.text_patches]
        self.rect_patches.clear()
        self.text_patches.clear()
        
        for res in self.roi_results:
            color = '#D9534F' if res["faulty"] else '#5CB85C'
            x1, y1, x2, y2 = res["coords"]
            
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4, edgecolor=color, facecolor='none', picker=True)
            self.ax.add_patch(rect)
            self.rect_patches.append(rect)
            
            text = self.ax.text(x1, y1 - 10, f"ROI {res['id']}: {'NG' if res['faulty'] else 'OK'}", 
                                color=color, fontsize=12, weight='bold', bbox=dict(facecolor='#FFFFFF', alpha=0.8))
            self.text_patches.append(text)
            
        self.canvas.draw()
        
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        cx, cy = event.xdata, event.ydata
        if cx is None or cy is None:
            return
            
        # Find which ROI rectangle was clicked
        for idx, res in enumerate(self.roi_results):
            x1, y1, x2, y2 = res["coords"]
            # Allow some margin for click precision
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                # Flip state!
                res["faulty"] = not res["faulty"]
                print(f"[Feedback] ROI {res['id']} manually changed to: {'FAULTY' if res['faulty'] else 'CLEAN'}")
                self.draw_rois()
                break
                
    def disconnect(self):
        self.canvas.mpl_disconnect(self.cid)
