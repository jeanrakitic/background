import tkinter as tk
import os
from tkinter import filedialog, ttk , messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
import torch 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")

        # Initialize all variables
        self.original_image = None
        self.processed_image = None
        self.thresholded_image = None
        self.file_path = None
        self.output_dir = os.path.join(os.getcwd(), "output_images")
        self.mask_dir = os.path.join(os.getcwd(), "u2net_results")
        self.model_path = os.path.join(os.getcwd(), "saved_models", "u2net", "u2net.pth")
        self.max_width = 1920
        self.max_height = 1080
        self.aspect_ratio = 1.0
        self.aspect_ratio_locked = tk.BooleanVar(value=True)
        self.updating_sliders = False
        self.threshold_value = tk.IntVar(value=127)
        self.max_value = tk.IntVar(value=255)
        self.comparison_value = tk.DoubleVar(value=50)

        # MediaPipe Selfie Segmentation
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        # Threshold types
        self.threshold_types = {
            "Binary": cv2.THRESH_BINARY,
            "Binary Inverted": cv2.THRESH_BINARY_INV,
            "Binary Thresholding": cv2.THRESH_BINARY,
            "Truncate": cv2.THRESH_TRUNC+cv2.THRESH_BINARY,
            "To Zero": cv2.THRESH_TOZERO,
            "To Zero Inverted": cv2.THRESH_TOZERO_INV,
            "Otsu" : cv2.THRESH_OTSU,
            "Adaptive":cv2.ADAPTIVE_THRESH_MEAN_C+cv2.THRESH_BINARY
        }
        self.current_threshold_type = tk.StringVar(value="Binary")

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Load Image Button
        self.load_btn = ttk.Button(self.main_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=0, column=0, pady=5)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create tabs
        self.preview_tab = ttk.Frame(self.notebook)
        self.resize_tab = ttk.Frame(self.notebook)
        self.threshold_tab = ttk.Frame(self.notebook)
        self.mediapipe_tab = ttk.Frame(self.notebook)
        self.comparison_tab = ttk.Frame(self.notebook)
        self.grabcut_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.preview_tab, text='Preview')
        self.notebook.add(self.resize_tab, text='Resize')
        self.notebook.add(self.threshold_tab, text='Threshold')
        self.notebook.add(self.comparison_tab, text='Compare')
        self.notebook.add(self.mediapipe_tab, text='Media Pipe')
        self.notebook.add(self.grabcut_tab, text="(GrabCut)")
        self.notebook.add(self.model_tab, text="Background Removal (Model)")

        

        # Setup tabs
        self.setup_preview_tab()
        self.setup_resize_tab()
        self.setup_threshold_tab()
        self.setup_comparison_tab()
        self.setup_mediapipe_tab()
        self.setup_grabcut_tab()
        self.setup_model_tab()
        

    # ---------- Tab Setup Methods ----------
    def setup_preview_tab(self):
        self.preview_label = ttk.Label(self.preview_tab, text="Original Image")
        self.preview_label.grid(row=0, column=0, pady=(5, 0))

        self.preview_canvas = tk.Canvas(self.preview_tab, width=400, height=400, bg='lightgray')
        self.preview_canvas.grid(row=1, column=0, padx=5, pady=5)

        self.resolution_label = ttk.Label(self.preview_tab, text="Resolution: -")
        self.resolution_label.grid(row=2, column=0, pady=5)

    def setup_model_tab(self):
        # Load Image Button
        controls_frame = ttk.LabelFrame(self.model_tab, text="model controls", padding="5")
        controls_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        apply_btn = ttk.Button(controls_frame, text="Apply model", command=self.run_background_removal)
        apply_btn.grid(row=0, column=0, pady=5)

        # Canvas for Image
        self.model_canvas = tk.Canvas(self.model_tab, width=300, height=300, bg="lightgray")
        self.model_result_canvas = tk.Canvas(self.model_tab, width=300, height=300, bg='lightgray')
        self.model_canvas.grid(row=1, column=0, padx=5, pady=5)
        self.model_result_canvas.grid(row=1, column=1, padx=5, pady=5)
        self.save_btn = ttk.Button(controls_frame, text="Save Image", command=self.save_image)
        self.save_btn.grid(row=3, column=0, columnspan=3, pady=10)

        # Remove Background Button
        

    def setup_grabcut_tab(self):
        # Load Image Button
        controls_frame = ttk.LabelFrame(self.grabcut_tab, text="grabcut controls", padding="5")
        controls_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        apply_btn = ttk.Button(controls_frame, text="Apply grabcut", command=self.run_grabcut)
        apply_btn.grid(row=0, column=0, pady=5)
        # Canvas for Original Image
        self.grabcut_canvas = tk.Canvas(self.grabcut_tab, width=300, height=300, bg="lightgray") 
        self.grabcut_result_canvas = tk.Canvas(self.grabcut_tab, width=300, height=300, bg='lightgray')
        self.grabcut_canvas.grid(row=1, column=0, padx=5, pady=5)
        self.grabcut_result_canvas.grid(row=1, column=1, padx=5, pady=5)
        self.save_btn = ttk.Button(controls_frame, text="Save Image", command=self.save_image)
        self.save_btn.grid(row=3, column=0, columnspan=3, pady=10)
        
        

        # Process Button
      
    

    def setup_resize_tab(self):
        # Resize controls frame
        self.resize_frame = ttk.LabelFrame(self.resize_tab, text="Resize Controls", padding="5")
        self.resize_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        # Aspect ratio lock checkbox
        self.aspect_ratio_check = ttk.Checkbutton(
            self.resize_frame,
            text="Lock Aspect Ratio",
            variable=self.aspect_ratio_locked,
            command=self.on_aspect_ratio_toggle
        )
        self.aspect_ratio_check.grid(row=0, column=0, columnspan=3, pady=5)

        # Width slider
        ttk.Label(self.resize_frame, text="Width:").grid(row=1, column=0, padx=5, pady=5)
        self.width_var = tk.IntVar()
        self.width_slider = ttk.Scale(
            self.resize_frame,
            from_=1,
            to=self.max_width,
            orient=tk.HORIZONTAL,
            variable=self.width_var,
            command=self.on_width_change
        )
        self.width_slider.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.width_label = ttk.Label(self.resize_frame, text="0")
        self.width_label.grid(row=1, column=2, padx=5)

        # Height slider
        ttk.Label(self.resize_frame, text="Height:").grid(row=2, column=0, padx=5, pady=5)
        self.height_var = tk.IntVar()
        self.height_slider = ttk.Scale(
            self.resize_frame,
            from_=1,
            to=self.max_height,
            orient=tk.HORIZONTAL,
            variable=self.height_var,
            command=self.on_height_change
        )
        self.height_slider.grid(row=2, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.height_label = ttk.Label(self.resize_frame, text="0")
        self.height_label.grid(row=2, column=2, padx=5)

        # Save button
        self.save_btn = ttk.Button(self.resize_frame, text="Save Resized Image", command=self.save_image)
        self.save_btn.grid(row=3, column=0, columnspan=3, pady=10)

        # Display areas for original and resized images
        ttk.Label(self.resize_tab, text="Original Image").grid(row=1, column=0, pady=(5, 0))
        self.original_canvas = tk.Canvas(self.resize_tab, width=300, height=300, bg='lightgray')
        self.original_canvas.grid(row=2, column=0, padx=5, pady=5)

        ttk.Label(self.resize_tab, text="Resized Image").grid(row=1, column=1, pady=(5, 0))
        self.resized_canvas = tk.Canvas(self.resize_tab, width=300, height=300, bg='lightgray')
        self.resized_canvas.grid(row=2, column=1, padx=5, pady=5)

    def on_width_change(self, _):
        if self.updating_sliders:
            return
            
        new_width = self.width_var.get()
        self.width_label.config(text=str(new_width))
        
        if self.aspect_ratio_locked.get() and self.original_image is not None:
            self.updating_sliders = True
            new_height = int(new_width / self.aspect_ratio)
            self.height_var.set(new_height)
            self.height_label.config(text=str(new_height))
            self.updating_sliders = False
            
        self.resize_image()


    def resize_image(self):
        if self.original_image is None:
            return
            
        new_width = self.width_var.get()
        new_height = self.height_var.get()
        
        # Perform the actual resize using cv2
        self.processed_image = cv2.resize(self.original_image, (new_width, new_height))
        
        # Display the resized image
        self.display_image(self.processed_image, self.resized_canvas, 300)

    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "No processed image to save! Please process an image first.")
            return
    
        # Open file dialog to save the image
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # Check if processed_image is a PIL.Image or a NumPy array
                if isinstance(self.processed_image, np.ndarray):
                    # Save directly if it's a NumPy array
                    cv2.imwrite(file_path, self.processed_image)
                elif isinstance(self.processed_image, Image.Image):
                    # Convert to NumPy array (if it's a PIL.Image) and save
                    processed_image_np = np.array(self.processed_image)
                    processed_image_np = cv2.cvtColor(processed_image_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(file_path, processed_image_np)
                else:
                    messagebox.showerror("Error", "Processed image format not recognized!")
                    return
    
                messagebox.showinfo("Success", f"Image successfully saved at {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save the image: {str(e)}")

    

    
    def on_height_change(self, _):
        if self.updating_sliders:
            return
            
        new_height = self.height_var.get()
        self.height_label.config(text=str(new_height))
        
        if self.aspect_ratio_locked.get() and self.original_image is not None:
            self.updating_sliders = True
            new_width = int(new_height * self.aspect_ratio)
            self.width_var.set(new_width)
            self.width_label.config(text=str(new_width))
            self.updating_sliders = False
            
        self.resize_image()

    def on_aspect_ratio_toggle(self):
        if self.aspect_ratio_locked.get() and self.original_image is not None:
            # When locking, adjust height to match current width's aspect ratio
            new_height = int(self.width_var.get() / self.aspect_ratio)
            self.height_var.set(new_height)
            self.height_label.config(text=str(new_height))
            self.resize_image()

    def setup_threshold_tab(self):
        controls_frame = ttk.LabelFrame(self.threshold_tab, text="Threshold Controls", padding="5")
        controls_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(controls_frame, text="Threshold Type:").grid(row=0, column=0, padx=5, pady=5)
        threshold_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.current_threshold_type,
            values=list(self.threshold_types.keys()),
            state="readonly"
        )
        threshold_combo.grid(row=0, column=1, padx=5, pady=5)
        threshold_combo.bind('<<ComboboxSelected>>', lambda e: self.apply_threshold())

        ttk.Label(controls_frame, text="Threshold Value:").grid(row=1, column=0, padx=5, pady=5)
        threshold_slider = ttk.Scale(
            controls_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.threshold_value,
            command=lambda _: self.apply_threshold()
        )
        threshold_slider.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        self.save_btn = ttk.Button(controls_frame, text="Save threshold", command=self.save_threshold_image)
        self.save_btn.grid(row=2, column=0, columnspan=2, pady=10)

        self.threshold_original_canvas = tk.Canvas(self.threshold_tab, width=300, height=300, bg='lightgray')
        self.threshold_result_canvas = tk.Canvas(self.threshold_tab, width=300, height=300, bg='lightgray')
        self.threshold_original_canvas.grid(row=1, column=0, padx=5, pady=5)
        self.threshold_result_canvas.grid(row=1, column=1, padx=5, pady=5)
        

    def apply_threshold(self):
        if self.original_image is None:
            return
            
        if len(self.original_image.shape) == 3:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original_image
            
        _, self.thresholded_image = cv2.threshold(
            gray,
            self.threshold_value.get(),
            self.max_value.get(),
            self.threshold_types[self.current_threshold_type.get()]
        )
        
        if len(self.original_image.shape) == 3:
            self.thresholded_image = cv2.cvtColor(self.thresholded_image, cv2.COLOR_GRAY2BGR)
            
        self.display_image(self.thresholded_image, self.threshold_result_canvas, 300)
        self.update_comparison()

        
    def save_threshold_image(self):
        if self.thresholded_image is None:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.thresholded_image)

    def update_comparison(self, *args):
        if self.original_image is None or self.thresholded_image is None:
            return
        canvas_width = self.comparison_canvas.winfo_width()
        canvas_height = self.comparison_canvas.winfo_height()
        split_position = int((self.comparison_value.get() / 100) * canvas_width)
        
        original_display = self.prepare_image_for_display(self.original_image, canvas_width, canvas_height)
        thresholded_display = self.prepare_image_for_display(self.thresholded_image, canvas_width, canvas_height)
        
        combined = np.copy(original_display)
        combined[:, split_position:] = thresholded_display[:, split_position:]
        
        photo = ImageTk.PhotoImage(Image.fromarray(combined))
        self.comparison_canvas.delete("all")
        self.comparison_canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
        self.comparison_canvas.image = photo
        self.comparison_canvas.create_line(split_position, 0, split_position, canvas_height, fill='white', width=2)

    def on_comparison_drag(self, event):
        canvas_width = self.comparison_canvas.winfo_width()
        value = max(0, min(100, (event.x / canvas_width) * 100))
        self.comparison_value.set(value)
        self.update_comparison()

    def prepare_image_for_display(self, image, canvas_width, canvas_height):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        scale = min(canvas_width/width, canvas_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image_rgb, (new_width, new_height))
        display = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        y_offset = (canvas_height - new_height) // 2
        x_offset = (canvas_width - new_width) // 2
        display[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        return display


    def setup_comparison_tab(self):
        self.comparison_frame = ttk.Frame(self.comparison_tab)
        self.comparison_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.comparison_canvas = tk.Canvas(self.comparison_frame, width=600, height=400, bg='lightgray')
        self.comparison_canvas.grid(row=0, column=0, padx=5, pady=5)

        self.comparison_slider = ttk.Scale(
            self.comparison_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.comparison_value,
            command=self.update_comparison
        )
        self.comparison_slider.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

    def setup_mediapipe_tab(self):
        controls_frame = ttk.LabelFrame(self.mediapipe_tab, text="MediaPipe Controls", padding="5")
        controls_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))

        apply_btn = ttk.Button(controls_frame, text="Apply MediaPipe", command=self.apply_mediapipe)
        apply_btn.grid(row=0, column=0, pady=5)

        self.mediapipe_original_canvas = tk.Canvas(self.mediapipe_tab, width=300, height=300, bg='lightgray')
        self.mediapipe_result_canvas = tk.Canvas(self.mediapipe_tab, width=300, height=300, bg='lightgray')
        self.mediapipe_original_canvas.grid(row=1, column=0, padx=5, pady=5)
        self.mediapipe_result_canvas.grid(row=1, column=1, padx=5, pady=5)
        self.save_btn = ttk.Button(controls_frame, text="Save Image", command=self.save_image)
        self.save_btn.grid(row=3, column=0, columnspan=3, pady=10)

    # ---------- MediaPipe Methods ----------
    def apply_mediapipe(self):
        """Apply MediaPipe Selfie Segmentation."""
        if self.original_image is None:
            return

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        result = self.segmenter.process(img_rgb)
        mask = result.segmentation_mask > 0.5

        # Replace background with white
        bg_color = (255, 255, 255)  # White background
        bg_image = np.zeros(self.original_image.shape, dtype=np.uint8)
        bg_image[:] = bg_color

        self.processed_image = np.where(mask[:, :, None], self.original_image, bg_image)
        self.display_image(self.processed_image, self.mediapipe_result_canvas, 300)
   

    
    def run_background_removal(self):
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return
    
        # Validate the file path
        if not self.file_path or not os.path.exists(self.file_path):
            messagebox.showerror("Error", "Invalid file path! Please load a valid image.")
            return
    
        try:
            # Debugging: Print file path
            print(f"Processing file: {self.file_path}")
    
            # Prepare dataset and model
            dataset = SalObjDataset(
                img_name_list=[self.file_path],
                lbl_name_list=[],
                transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
            )
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
            net = U2NET(3, 1)
            net.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            net.eval()
    
            # Process image
            for data in dataloader:
                inputs = Variable(data["image"].type(torch.FloatTensor))
                d1, *_ = net(inputs)
                pred = (d1[:, 0, :, :] - torch.min(d1)) / (torch.max(d1) - torch.min(d1))
    
                predict = pred.squeeze().cpu().data.numpy()
                mask = Image.fromarray((predict * 255).astype(np.uint8)).convert("L")
    
                original_image = Image.open(self.file_path).convert("RGBA")
                mask = mask.resize(original_image.size, Image.BILINEAR)
    
                transparent_bg = Image.new("RGBA", original_image.size, (255, 255, 255, 0))
                final_image = Image.composite(original_image, transparent_bg, mask)
    
                # Store processed image
                self.processed_image = final_image
    
                # Display processed image
                final_image_resized = final_image.resize((300, 300))
                final_image_tk = ImageTk.PhotoImage(final_image_resized)
                self.model_result_canvas.delete("all")
                self.model_result_canvas.create_image(150, 150, image=final_image_tk, anchor=tk.CENTER)
                self.model_result_canvas.image = final_image_tk
    
                messagebox.showinfo("Success", "Background removed and displayed.")
    
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    

    

    def run_grabcut(self):
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return
    
        try:
            # Create a mask and initialize GrabCut models
            mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            rect = (10, 10, self.original_image.shape[1] - 20, self.original_image.shape[0] - 20)
            bg_model = np.zeros((1, 65), np.float64)
            fg_model = np.zeros((1, 65), np.float64)
    
            # Apply GrabCut algorithm
            cv2.grabCut(self.original_image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    
            # Generate the final mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
            result = self.original_image * mask2[:, :, np.newaxis]
    
            # Create an output image with a white background
            white_background = self.original_image.copy()
            white_background[mask2 == 0] = [255, 255, 255]
    
            # Store the processed image for saving
            self.processed_image = white_background
    
            # Convert the processed image to a format suitable for display
            result_image = Image.fromarray(cv2.cvtColor(white_background, cv2.COLOR_BGR2RGB))
            result_resized = result_image.resize((300, 300))  # Resize to fit canvas
            result_tk = ImageTk.PhotoImage(result_resized)
    
            # Display the processed image in the GrabCut result canvas
            self.grabcut_result_canvas.delete("all")
            self.grabcut_result_canvas.create_image(150, 150, image=result_tk, anchor=tk.CENTER)
            self.grabcut_result_canvas.image = result_tk  # Save reference to prevent garbage collection
    
            # Notify the user
            messagebox.showinfo("Success", "Background removed using GrabCut and displayed in the result canvas.")
    
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            # Store the file path for later use
            self.file_path = file_path
    
            # Load the image using OpenCV
            self.original_image = cv2.imread(file_path)
    
            if self.original_image is not None:
                # Calculate aspect ratio
                height, width = self.original_image.shape[:2]
                self.aspect_ratio = width / height
    
                # Update all displays
                self.display_image(self.original_image, self.preview_canvas, 400)
                self.display_image(self.original_image, self.original_canvas, 300)
                self.display_image(self.original_image, self.mediapipe_original_canvas, 300)
                self.display_image(self.original_image, self.threshold_original_canvas, 300)
                self.display_image(self.original_image, self.model_canvas, 300)
                self.display_image(self.original_image, self.grabcut_canvas, 300)
    
                # Update resolution labels
                self.resolution_label.config(text=f"Resolution: {width}x{height}")
    
                # Update sliders
                self.width_var.set(width)
                self.height_var.set(height)
                self.width_label.config(text=str(width))
                self.height_label.config(text=str(height))
    
            else:
                # Handle invalid image load
                messagebox.showerror("Error", "Failed to load the image. Please select a valid image file.")
        else:
            # Handle no file selected
            messagebox.showerror("Error", "No file selected!")


    def display_image(self, image, canvas, max_size):
        """Display an image on a given canvas with resizing."""
        if image is None:
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image_rgb, (new_width, new_height))
        photo = ImageTk.PhotoImage(Image.fromarray(resized_image))
        canvas.delete("all")
        canvas.create_image(max_size // 2, max_size // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()
