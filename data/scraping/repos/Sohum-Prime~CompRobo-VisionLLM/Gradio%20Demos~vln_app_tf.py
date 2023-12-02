import gradio as gr
import tensorflow as tf
import openai
from PIL import Image
from io import BytesIO

# Load a pre-trained TensorFlow model (MobileNetV2 in this case)
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))


# Function to preprocess the image for MobileNetV2
def preprocess_image(image):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, 0)
    return image


# Function to get image predictions
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    labels = tf.keras.applications.mobilenet_v2.decode_predictions(prediction)
    return labels[0][0][1]


# Function to generate text response using GPT-3.5 Turbo
def generate_text(prompt):
    openai.api_key = "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"  # Replace with your OpenAI API key
    response = openai.Completion.create(
        model="text-davinci-003",  # GPT-3.5 Turbo model
        prompt=prompt,
        max_tokens=2000,
        temperature=0.7,
        n=1,
        stop=None,
        logprobs=None,
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
    inputs=[
        gr.Image(label="Upload Image", type="pil"),
        gr.Textbox(label="Enter your instruction"),
    ],
    outputs="text",
    title="Vision and Language Navigation Demo",
    description="This app detects objects in images and generates responses based on user instructions.",
)

# Run the app
iface.launch()
