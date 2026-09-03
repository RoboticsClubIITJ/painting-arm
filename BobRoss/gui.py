import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np

# Import the necessary functions from your existing cv.py
from cv import preprocess_image, extract_smoothed_contours, map_paths_to_workspace

class XDoGGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("XDoG Contour Generator")
        self.raw_image = None
        self.cv_paths = None

        # Control Panel
        control_frame = tk.Frame(master, width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=15)

        tk.Button(control_frame, text="Upload Image", command=self.load_image, width=20, bg="#4CAF50", fg="white").pack(pady=10)

        # XDoG Variables
        self.sigma = tk.DoubleVar(value=1.4)
        self.k_sigma = tk.DoubleVar(value=1.6)
        self.epsilon = tk.DoubleVar(value=0.01)
        self.phi = tk.IntVar(value=20)
        self.gamma = tk.DoubleVar(value=0.98)

        # Sliders
        self.create_slider(control_frame, "Sigma", self.sigma, 0.1, 5.0, 0.1)
        self.create_slider(control_frame, "Sigma-k", self.k_sigma, 1.0, 5.0, 0.1)
        self.create_slider(control_frame, "Epsilon", self.epsilon, 0.001, 0.1, 0.001)
        self.create_slider(control_frame, "Phi", self.phi, 1, 100, 1)
        self.create_slider(control_frame, "Gamma", self.gamma, 0.5, 1.0, 0.01)

        tk.Button(control_frame, text="Confirm & Send to Robot", command=self.confirm, width=20, bg="#2196F3", fg="white").pack(pady=20)

        # Image Preview Panel
        self.canvas_width = 800
        self.canvas_height = 800
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg='#333333')
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.img_on_canvas = None

    def create_slider(self, parent, label, var, min_val, max_val, res):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        tk.Label(frame, text=label).pack(side=tk.LEFT)
        scale = tk.Scale(frame, variable=var, from_=min_val, to=max_val, resolution=res, 
                         orient=tk.HORIZONTAL, command=self.update_preview)
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.raw_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.update_preview()

    def update_preview(self, *args):
        if self.raw_image is None:
            return

        # Process image with current slider values
        binary_sketch = preprocess_image(
            self.raw_image,
            sigma=self.sigma.get(),
            k_sigma=self.k_sigma.get(),
            epsilon=self.epsilon.get(),
            phi=self.phi.get(),
            gamma=self.gamma.get()
        )

        # Convert back to format tkinter can display
        preview = Image.fromarray(binary_sketch)
        preview.thumbnail((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(preview)

        # Update canvas
        if self.img_on_canvas is None:
            self.img_on_canvas = self.canvas.create_image(
                self.canvas_width//2, self.canvas_height//2, 
                image=self.tk_image, anchor=tk.CENTER
            )
        else:
            self.canvas.itemconfig(self.img_on_canvas, image=self.tk_image)

    def confirm(self):
        if self.raw_image is not None:
            binary_sketch = preprocess_image(
                self.raw_image,
                sigma=self.sigma.get(),
                k_sigma=self.k_sigma.get(),
                epsilon=self.epsilon.get(),
                phi=self.phi.get(),
                gamma=self.gamma.get()
            )
            paths = extract_smoothed_contours(binary_sketch)
            self.cv_paths = map_paths_to_workspace(paths, self.raw_image.shape)
        
        self.master.quit()

def run_gui():
    root = tk.Tk()
    app = XDoGGUI(root)
    root.mainloop()
    paths = app.cv_paths
    root.destroy()
    return paths

if __name__ == "__main__":
    run_gui()