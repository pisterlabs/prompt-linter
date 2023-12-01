import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import openai

# Load a pre-trained PyTorch model (Vision Transformer in this case)
model = models.vit_b_16(pretrained=True)
model.eval()

# Function to preprocess the image for Vision Transformer
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to get image predictions
def predict_image(image):
    processed_image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(processed_image)
    _, predicted = outputs.max(1)
    labels = predicted.item()
    return labels

# Function to generate text response using GPT-3.5 Turbo
def generate_text(prompt):
    openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your OpenAI API key
    response = openai.Completion.create(
        model="text-davinci-003",  # GPT-3.5 Turbo model
        prompt=prompt,
        max_tokens=600,  # Adjusted for ~200 words
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Gradio app interface
def vln_app(image, text):
    object_detected = predict_image(image)
    combined_prompt = f"Object detected: {object_detected}. User instruction: {text}"
    response = generate_text(combined_prompt)
    return response

# Create the Gradio interface
iface = gr.Interface(
    fn=vln_app,
    inputs=[gr.Image(label="Upload Image", type="pil"), gr.Textbox(label="Enter your instruction")],
    outputs=gr.Textbox(label="Response", lines=10),  # Adjusted for larger output
    title="Vision and Language Navigation Demo",
    description="This app detects objects in images and generates responses based on user instructions."
)

# Run the app
iface.launch()
