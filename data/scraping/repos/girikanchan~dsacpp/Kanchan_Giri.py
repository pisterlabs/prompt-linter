import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import requests
import torch
from torchvision.transforms import functional as F
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import os
import openai

openai.api_key = os.getenv("openai_key")

# Load the pre-trained DETR model
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

# Set up the Tkinter window
window = tk.Tk()
window.title("Recipe Detection")
window.geometry("400x400")

# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Open and display the selected image
            image = Image.open(file_path)
            image = image.resize((300, 300))  # Resize for display purposes
            img_tk = ImageTk.PhotoImage(image)
            image_label.configure(image=img_tk)
            image_label.image = img_tk

            # Process the image with DETR model
            input_image = F.to_tensor(image)
            input_image = F.normalize(input_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            inputs = feature_extractor(images=input_image.unsqueeze(0), return_tensors="pt")
            outputs = model(**inputs)

            # Convert outputs (bounding boxes and class logits) to COCO API
            # Let's only keep detections with score > 0.9
            target_sizes = torch.tensor([inputs['pixel_values'].shape[-2:]])
            results = feature_extractor.post_process(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            # Generate recipe descriptions using OpenAI API
            recipe_descriptions = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                recipe_descriptions.append(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

            # Display recipe descriptions
            recipe_text.delete(1.0, tk.END)  # Clear previous text
            recipe_text.insert(tk.END, "\n".join(recipe_descriptions))

        except Exception as e:
            messagebox.showerror("Error", str(e))

# Image display label
image_label = tk.Label(window)
image_label.pack()

# Upload button
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack()

# Recipe text box
recipe_text = tk.Text(window, height=10)
recipe_text.pack()

# Run the Tkinter event loop
window.mainloop()
