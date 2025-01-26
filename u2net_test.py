import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET

class CombinedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Combined Image Processor")

        # Initialize variables
        self.original_image = None
        self.thresholded_image = None
        self.file_path = None
        self.output_dir = os.path.join(os.getcwd(), "output_images")
        self.mask_dir = os.path.join(os.getcwd(), "u2net_results")
        self.model_path = os.path.join(os.getcwd(), "saved_models", "u2net", "u2net.pth")

        self.threshold_value = tk.IntVar(value=127)
        self.max_value = tk.IntVar(value=255)
        self.threshold_types = {
            "Binary": cv2.THRESH_BINARY,
            "Binary Inverted": cv2.THRESH_BINARY_INV,
            "Truncate": cv2.THRESH_TRUNC,
            "To Zero": cv2.THRESH_TOZERO,
            "To Zero Inverted": cv2.THRESH_TOZERO_INV
        }
        self.current_threshold_type = tk.StringVar(value="Binary")

        self.setup_ui()

    def setup_ui(self):
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self.model_tab = ttk.Frame(self.notebook)
        self.grabcut_tab = ttk.Frame(self.notebook)
        self.threshold_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.model_tab, text="Background Removal (Model)")
        self.notebook.add(self.grabcut_tab, text="Background Removal (GrabCut)")
        self.notebook.add(self.threshold_tab, text="Thresholding")

        self.setup_model_tab()
        self.setup_grabcut_tab()
        self.setup_threshold_tab()

    def setup_model_tab(self):
        # Load Image Button
        ttk.Button(self.model_tab, text="Load Image", command=self.load_image_model).pack(pady=5)

        # Canvas for Image
        self.model_canvas = tk.Canvas(self.model_tab, width=400, height=400, bg="lightgray")
        self.model_canvas.pack(pady=10)

        # Remove Background Button
        ttk.Button(self.model_tab, text="Remove Background", command=self.run_background_removal).pack(pady=5)

    def setup_grabcut_tab(self):
        # Load Image Button
        ttk.Button(self.grabcut_tab, text="Load Image", command=self.load_image_grabcut).pack(pady=5)

        # Canvas for Original Image
        self.grabcut_canvas = tk.Canvas(self.grabcut_tab, width=400, height=400, bg="lightgray")
        self.grabcut_canvas.pack(pady=10)

        # Process Button
        ttk.Button(self.grabcut_tab, text="Remove Background", command=self.run_grabcut).pack(pady=5)

    def setup_threshold_tab(self):
        # Main frame
        main_frame = tk.Frame(self.threshold_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image Display Canvases
        self.original_canvas = tk.Canvas(main_frame, width=300, height=300, bg="lightgray")
        self.original_canvas.grid(row=0, column=0, padx=10, pady=10)

        self.result_canvas = tk.Canvas(main_frame, width=300, height=300, bg="lightgray")
        self.result_canvas.grid(row=0, column=1, padx=10, pady=10)

        # Controls Frame
        controls_frame = tk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, columnspan=2, pady=10)

        # Threshold Type Dropdown
        tk.Label(controls_frame, text="Threshold Type:").grid(row=0, column=0, padx=5, pady=5)
        threshold_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.current_threshold_type,
            values=list(self.threshold_types.keys()),
            state="readonly"
        )
        threshold_combo.grid(row=0, column=1, padx=5, pady=5)
        threshold_combo.bind('<<ComboboxSelected>>', lambda e: self.apply_threshold())

        # Threshold Value Slider
        tk.Label(controls_frame, text="Threshold Value:").grid(row=1, column=0, padx=5, pady=5)
        threshold_slider = ttk.Scale(
            controls_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.threshold_value,
            command=lambda _: self.apply_threshold()
        )
        threshold_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Load Image Button
        load_button = tk.Button(controls_frame, text="Load Image", command=self.load_image_threshold)
        load_button.grid(row=2, column=0, padx=5, pady=5)

        # Save Thresholded Image Button
        save_button = tk.Button(controls_frame, text="Save Thresholded Image", command=self.save_threshold_image)
        save_button.grid(row=2, column=1, padx=5, pady=5)

    def load_image_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.file_path = file_path
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, self.model_canvas)

    def load_image_grabcut(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            self.file_path = file_path
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, self.grabcut_canvas)

    def load_image_threshold(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.file_path = file_path
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, self.original_canvas)

    def apply_threshold(self):
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, self.thresholded_image = cv2.threshold(
            gray,
            self.threshold_value.get(),
            self.max_value.get(),
            self.threshold_types[self.current_threshold_type.get()]
        )
        self.display_image(self.thresholded_image, self.result_canvas)

    def save_threshold_image(self):
        if self.thresholded_image is None:
            messagebox.showerror("Error", "No thresholded image to save!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.thresholded_image)
            messagebox.showinfo("Saved", f"Thresholded image saved to {file_path}")

    def run_background_removal(self):
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        try:
            os.makedirs(self.mask_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)

            dataset = SalObjDataset(
                img_name_list=[self.file_path],
                lbl_name_list=[],
                transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
            )

            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
            net = U2NET(3, 1)
            net.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            net.eval()

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

                output_path = os.path.join(self.output_dir, "output.png")
                final_image.save(output_path)

                final_image_resized = final_image.resize((400, 400))
                final_image_tk = ImageTk.PhotoImage(final_image_resized)
                self.model_canvas.delete("all")
                self.model_canvas.create_image(200, 200, image=final_image_tk, anchor=tk.CENTER)
                self.model_canvas.image = final_image_tk

                messagebox.showinfo("Success", f"Background removed and saved at {output_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_grabcut(self):
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        try:
            mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            rect = (10, 10, self.original_image.shape[1] - 20, self.original_image.shape[0] - 20)
            bg_model = np.zeros((1, 65), np.float64)
            fg_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(self.original_image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
            result = self.original_image * mask2[:, :, np.newaxis]

            white_background = self.original_image.copy()
            white_background[mask2 == 0] = [255, 255, 255]

            output_path = os.path.join(self.output_dir, "grabcut_result.png")
            cv2.imwrite(output_path, white_background)

            result_image = Image.fromarray(cv2.cvtColor(white_background, cv2.COLOR_BGR2RGB))
            result_resized = result_image.resize((400, 400))
            result_tk = ImageTk.PhotoImage(result_resized)

            self.grabcut_canvas.delete("all")
            self.grabcut_canvas.create_image(200, 200, image=result_tk, anchor=tk.CENTER)
            self.grabcut_canvas.image = result_tk

            messagebox.showinfo("Success", f"Background removed using GrabCut and saved at {output_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_image(self, image, canvas):
        if image is None:
            return

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        scale = min(400 / height, 400 / width)
        resized = cv2.resize(image_rgb, (int(width * scale), int(height * scale)))
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        canvas.delete("all")
        canvas.create_image(200, 200, image=photo, anchor=tk.CENTER)
        canvas.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = CombinedApp(root)
    root.mainloop()
