import sys
import os
import random
import subprocess
import importlib.util
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QTextEdit, QLabel, QMessageBox, QFileDialog, QFrame, QInputDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Helper to import files that start with numbers
def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_dir = os.path.dirname(os.path.abspath(__file__))
utils = import_script("utils", os.path.join(base_dir, "utils.py"))
step02 = import_script("step02", os.path.join(base_dir, "02_select_rois.py"))
step07 = import_script("step07", os.path.join(base_dir, "07_predict.py"))
step08 = import_script("step08", os.path.join(base_dir, "08_manual_feedback.py"))

PIPELINE_STEPS = {
    1: ("Inspect Data", "01_load_and_inspect.py"),
    2: ("Select ROIs (Embedded)", "02_select_rois.py"),
    3: ("Extract ROIs", "03_extract_rois.py"),
    4: ("Process CV Features", "04_process_rois.py"),
    5: ("Train Models", "05_train_roi_classifier.py"),
    6: ("Classify Full Images", "06_classify_images.py"),
    7: ("Predict Image (Embedded)", "07_predict.py"),
    8: ("Manual Labeling (Embedded)", "08_manual_feedback.py"),
    9: ("Supervised Retraining", "09_retrain_supervised.py"),
    10: ("Export Results Metadata", "10_export_results.py")
}

# --- STYLESHEET (Professional Cream & Peach Theme) ---
APP_STYLE = """
QMainWindow {
    background-color: #FFF0E5;
}
QLabel {
    color: #5C4033;
    font-weight: bold;
}
QPushButton {
    background-color: #FFCCB6;
    color: #5C4033;
    border: none;
    border-radius: 8px;
    padding: 10px;
    font-weight: bold;
    text-align: left;
    padding-left: 15px;
}
QPushButton:hover {
    background-color: #FFB39A;
}
QPushButton:pressed {
    background-color: #FF9A76;
    color: #FFFFFF;
}
QPushButton:disabled {
    background-color: #FFEBDF;
    color: #A0A0A0;
}
QTextEdit {
    background-color: #FFFFFF;
    color: #333333;
    border: 2px solid #FFCCB6;
    border-radius: 8px;
    padding: 10px;
    font-family: monospace;
}
QFrame#canvas_container {
    background-color: #FFFFFF;
    border: 2px solid #FFCCB6;
    border-radius: 8px;
}
"""

class ScriptRunner(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int, bool)

    def __init__(self, step_number, cmd):
        super().__init__()
        self.step_number = step_number
        self.cmd = cmd

    def run(self):
        try:
            if self.cmd[0].endswith("python") or self.cmd[0].endswith("python3"):
                if "-u" not in self.cmd:
                    self.cmd.insert(1, "-u")
            
            process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in process.stdout:
                self.output_signal.emit(line)
                
            process.stdout.close()
            return_code = process.wait()
            self.finished_signal.emit(self.step_number, return_code == 0)
            
        except Exception as e:
            self.output_signal.emit(f"ERROR: {str(e)}\n")
            self.finished_signal.emit(self.step_number, False)


class PCBPipelineApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB AI Pipeline | Pro Edition")
        self.resize(1200, 800)
        self.setStyleSheet(APP_STYLE)
        
        self.active_runner = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Embedded Plotting State
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = None
        
        self.roi_selector_obj = None
        self.feedback_selector_obj = None
        self.feedback_images = []
        self.current_feedback_idx = 0
        self.feedback_data = {}
        
        self.initUI()
        self.append_output("System Initialized. Ready to process PCB data.\n")
        
    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # --- LEFT PANEL (Controls) ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        
        title_label = QLabel("ANOMALY PIPELINE")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #FF9A76; margin-bottom: 10px;")
        left_panel.addWidget(title_label)
        
        self.buttons = {}
        for step, (name, _) in PIPELINE_STEPS.items():
            btn = QPushButton(f"0{step}. {name}" if step < 10 else f"{step}. {name}")
            btn.setFont(QFont("Arial", 11))
            btn.clicked.connect(lambda checked, s=step: self.run_step(s))
            self.buttons[step] = btn
            left_panel.addWidget(btn)
            
        left_panel.addStretch()
        
        # Save selection button specific for Step 2
        self.btn_save_rois = QPushButton("Save Embedded ROI Selection")
        self.btn_save_rois.setStyleSheet("background-color: #A0D8B3; color: #5C4033;")
        self.btn_save_rois.clicked.connect(self.save_embedded_rois)
        self.btn_save_rois.hide()
        left_panel.addWidget(self.btn_save_rois)

        # Next button specific for Step 8
        self.btn_next_feedback = QPushButton("Save & Next Image ->")
        self.btn_next_feedback.setStyleSheet("background-color: #A0D8B3; color: #5C4033;")
        self.btn_next_feedback.clicked.connect(self.next_feedback_image)
        self.btn_next_feedback.hide()
        left_panel.addWidget(self.btn_next_feedback)
        
        left_container = QWidget()
        left_container.setLayout(left_panel)
        left_container.setFixedWidth(300)
        
        # --- RIGHT PANEL (Visuals & Output) ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        
        # Top: Matplotlib Canvas
        canvas_container = QFrame()
        canvas_container.setObjectName("canvas_container")
        canvas_layout = QVBoxLayout(canvas_container)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #FFCCB6; color: #5C4033;")
        canvas_layout.addWidget(self.toolbar)
        
        canvas_layout.addWidget(self.canvas)
        right_panel.addWidget(canvas_container, 3) # 60% height
        
        # Bottom: Console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        right_panel.addWidget(self.console, 2) # 40% height
        
        layout.addWidget(left_container)
        layout.addLayout(right_panel)
        
        # Init blank canvas
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        self.ax.text(0.5, 0.5, "Awaiting Visual Data...", ha='center', va='center', color='gray')
        self.canvas.draw()
        
    def append_output(self, text):
        self.console.moveCursor(self.console.textCursor().End)
        self.console.insertPlainText(text)
        self.console.moveCursor(self.console.textCursor().End)
        
    def set_buttons_enabled(self, enabled):
        for btn in self.buttons.values():
            btn.setEnabled(enabled)

    # --- STEP 2 HANDLING ---
    
    def handle_step_2_embedded(self):
        self.append_output("--- STARTED STEP 2: Embedded Interactive Selection ---\n")
        self.btn_save_rois.show()
        self.set_buttons_enabled(False)
        
        utils.ensure_dirs()
        metadata_path = os.path.join(utils.OUTPUT_DIR, "metadata.json")
        metadata = utils.load_json(metadata_path)
        
        if not metadata:
            self.append_output("[ERROR] metadata.json not found. Run Step 1 first.\n")
            self.on_step_finished(2, False)
            self.btn_save_rois.hide()
            return
            
        image_names = list(metadata.keys())
        ok_images = [name for name, data in metadata.items() if data['binary_label'] == "clean"]
        target_image_name = random.choice(ok_images) if ok_images else random.choice(image_names)
        target_image_path = metadata[target_image_name]['path']
        
        self.append_output(f"Loading image for ROI selection: {target_image_name}\n")
        
        img = utils.load_image(target_image_path)
        
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(img)
        self.ax.set_title(f"ROI Selection: {target_image_name}\nDraw boxes. Click 'Save' when done.", color='#5C4033')
        self.figure.patch.set_facecolor('#FFF0E5')
        self.ax.tick_params(colors='#5C4033')
        
        self.roi_selector_obj = step02.ROISelector(self.ax, target_image_name)
        self.canvas.draw()
        
    def save_embedded_rois(self):
        if not self.roi_selector_obj:
            return
            
        success = False
        if self.roi_selector_obj.rois:
            output_data = {
                "demo_image": self.roi_selector_obj.image_name,
                "rois": []
            }
            for idx, (x1, y1, x2, y2) in enumerate(self.roi_selector_obj.rois):
                output_data["rois"].append({"roi_id": idx + 1, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                
            roi_output_path = os.path.join(utils.OUTPUT_DIR, "roi_coordinates.json")
            utils.save_json(roi_output_path, output_data)
            self.append_output(f"\nSaved {len(self.roi_selector_obj.rois)} ROIs successfully.\n")
            success = True
        else:
            self.append_output("\nNo ROIs selected.\n")
            
        self.roi_selector_obj = None
        self.btn_save_rois.hide()
        
        self.figure.clear()
        self.canvas.draw()
        self.on_step_finished(2, success)

    # --- STEP 7 HANDLING ---
    
    def handle_step_7_embedded(self):
        self.append_output("\n--- STARTED STEP 7: Single Image Predict ---\n")
        
        data_dir = os.path.abspath(os.path.join(self.base_dir, '..', 'data_pro'))
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image for Prediction", data_dir, "Images (*.png *.jpg);;All Files (*)", options=options
        )
        if not file_name:
            self.append_output("\n[SYSTEM] Step 7 cancelled by user.\n")
            return
            
        img_rgb, img_is_faulty, roi_results, output_log = step07.predict_image(file_name)
        self.append_output(output_log)
        
        if img_rgb is not None:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.imshow(img_rgb)
            
            title_color = '#D9534F' if img_is_faulty else '#5CB85C'
            self.ax.set_title(f"Prediction: {'FAULTY (NG)' if img_is_faulty else 'CLEAN (OK)'}", color=title_color, weight='bold')
            self.figure.patch.set_facecolor('#FFF0E5')
            self.ax.tick_params(colors='#5C4033')
            
            for res in roi_results:
                color = '#D9534F' if res["faulty"] else '#5CB85C'
                x1, y1, x2, y2 = res["coords"]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor=color, facecolor='none')
                self.ax.add_patch(rect)
                self.ax.text(x1, y1 - 10, f"ROI {res['id']}: {'NG' if res['faulty'] else 'OK'}", 
                        color=color, fontsize=12, weight='bold', bbox=dict(facecolor='#FFFFFF', alpha=0.7))
            
            self.canvas.draw()
        
        self.on_step_finished(7, True)

    # --- STEP 8: MANUAL LABELING (ACTIVE LEARNING) ---
    
    def handle_step_8_embedded(self):
        reply = QMessageBox.question(self, "Manual Training", "Do you want to manually train the model on select images?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            self.append_output("\n[SYSTEM] Manual training skipped by user.\n")
            self.on_step_finished(8, True)
            return
            
        count, ok = QInputDialog.getInt(self, "Image Count", "How many images to manually review?", 5, 1, 100, 1)
        if not ok:
            return

        choice = QMessageBox.question(self, "Selection Method", "Do you want to select images manually?\n(Click YES for Manual Selection, NO for Random Selection)", QMessageBox.Yes | QMessageBox.No)
        
        metadata_path = os.path.join(utils.OUTPUT_DIR, "metadata.json")
        metadata = utils.load_json(metadata_path)
        all_images = [info["path"] for info in metadata.values()]

        if choice == QMessageBox.Yes:
            data_dir = os.path.abspath(os.path.join(self.base_dir, '..', 'data_pro'))
            files, _ = QFileDialog.getOpenFileNames(self, f"Select {count} Images", data_dir, "Images (*.png *.jpg)")
            if not files:
                return
            self.feedback_images = files[:count]
        else:
            self.feedback_images = random.sample(all_images, min(count, len(all_images)))

        self.current_feedback_idx = 0
        self.feedback_data = {}
        
        self.append_output(f"\n--- MANUAL TRAINING: Reviewing {len(self.feedback_images)} images ---\n")
        self.set_buttons_enabled(False)
        self.btn_next_feedback.show()
        
        self.load_feedback_image()
        
    def load_feedback_image(self):
        if self.current_feedback_idx >= len(self.feedback_images):
            # Finished Review Session!
            self.btn_next_feedback.hide()
            out_path = os.path.join(utils.OUTPUT_DIR, "feedback_labels.json")
            utils.save_json(out_path, self.feedback_data)
            self.append_output(f"\n[SUCCESS] Manually reviewed {len(self.feedback_data)} images. Overrides saved to {out_path}\n")
            
            self.figure.clear()
            self.canvas.draw()
            self.on_step_finished(8, True)
            return

        img_path = self.feedback_images[self.current_feedback_idx]
        self.append_output(f"Reviewing [{self.current_feedback_idx+1}/{len(self.feedback_images)}]: {os.path.basename(img_path)}\n")
        
        # Get baseline prediction to show first
        img_rgb, img_is_faulty, roi_results, _ = step07.predict_image(img_path)
        if img_rgb is None:
            self.append_output("Failed to load image. Skipping...\n")
            self.next_feedback_image()
            return
            
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(img_rgb)
        self.ax.set_title(f"Review: {os.path.basename(img_path)}\nCLICK any ROI to flip its state from OK to NG (or vice versa)", color='#5C4033', weight='bold')
        self.figure.patch.set_facecolor('#FFF0E5')
        self.ax.tick_params(colors='#5C4033')

        # Hook it into the feedback click-listener system
        self.feedback_selector_obj = step08.FeedbackSelector(self.canvas, self.ax, img_path, roi_results)
        
    def next_feedback_image(self):
        if self.feedback_selector_obj:
            self.feedback_data[self.feedback_selector_obj.image_name] = self.feedback_selector_obj.roi_results
            self.feedback_selector_obj.disconnect()
            self.feedback_selector_obj = None
            
        self.current_feedback_idx += 1
        self.load_feedback_image()

    # --- GENERAL RUNNER ---
    
    def run_step(self, step_number):
        if self.active_runner and self.active_runner.isRunning():
            QMessageBox.warning(self, "Warning", "A step is currently running. Please wait.")
            return

        name, script = PIPELINE_STEPS[step_number]
        
        # Check Embedded Overrides
        if step_number == 2:
            self.handle_step_2_embedded()
            return
        elif step_number == 7:
            self.handle_step_7_embedded()
            return
        elif step_number == 8:
            self.handle_step_8_embedded()
            return
            
        # Standard Subprocess (1, 3, 4, 5, 6, 9, 10)
        script_path = os.path.join(self.base_dir, script)
        cmd = [sys.executable, script_path]

        self.append_output(f"\n--- STARTED STEP {step_number}: {name} ---\n")
        self.set_buttons_enabled(False)
        
        self.active_runner = ScriptRunner(step_number, cmd)
        self.active_runner.output_signal.connect(self.append_output)
        self.active_runner.finished_signal.connect(self.on_step_finished)
        self.active_runner.start()
        
    def on_step_finished(self, step_number, success):
        self.set_buttons_enabled(True)
        self.append_output(f"\n--- FINISHED STEP {step_number} ---\n")
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())
        
        if success and step_number < 10:
            QTimer.singleShot(500, lambda: self.prompt_next_step(step_number))
            
    def prompt_next_step(self, step_number):
        reply = QMessageBox.question(
            self, 
            "Step Completed", 
            f"Step {step_number} successfully finished.\nWould you like to automatically start Step {step_number + 1}?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.Yes
        )
        if reply == QMessageBox.Yes:
            self.run_step(step_number + 1)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = PCBPipelineApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
