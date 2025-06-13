import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load your saved models
model = joblib.load("cat_dog_svm.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# HOG parameters (should match training time)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys"
}

IMG_SIZE = (128, 128)  # Resize images to this size

def extract_hog_features(img):
    return hog(img,
               orientations=HOG_PARAMS["orientations"],
               pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
               cells_per_block=HOG_PARAMS["cells_per_block"],
               block_norm=HOG_PARAMS["block_norm"])

def prepare_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        features = extract_hog_features(img)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        return features_pca
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

class CatDogClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐾 Cat vs Dog Classifier")

        self.label = tk.Label(root, text="Upload a Picture to Classify as Dog or Cat", font=("Arial", 14))
        self.label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="Select Image", command=self.upload_image,
                                    font=("Arial", 12), bg="blue", fg="white")
        self.upload_btn.pack(pady=15)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        # Display image
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img

        # Predict
        processed = prepare_image(file_path)
        if processed is not None:
            prediction = model.predict(processed)[0]
            result = "🐱 Cat" if prediction == 0 else "🐶 Dog"
            self.result_label.config(text=f"Prediction: {result}")
        else:
            self.result_label.config(text="⚠️ Invalid image or processing error.")

if __name__ == "__main__":
    root = tk.Tk()
    app = CatDogClassifierApp(root)
    root.mainloop()
