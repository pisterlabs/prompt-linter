import os 
import streamlit as st 
import openai
from PIL import Image
import requests
from io import BytesIO
from pages.utils.style import add_style 
from pages.utils.gen_app import generated_app, create_app

add_style()

# Set up the OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = f"{str(os.getenv('OPENAI_API_BASE'))}" 
openai.api_version = "2023-06-01-preview"
openai.api_type = "azure"

# Define the Streamlit app
def app():
    st.title("DALL·E 2 Image Generator")
    # Get the user prompt from the user
    prompt = st.text_input("Enter a prompt:")

    # Call the OpenAI API to generate an image
    if prompt:
        response = openai.Image.create(
            prompt=prompt,
            size='1024x1024',
            n=1
        )

        # Display the resulting image
        image_url = response["data"][0]["url"]
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, width=500, use_column_width=True, caption=prompt)
        
        # Add buttons for copy prompt, regenerate and download image
        col1, col2, col3 = st.columns(3)
        if col1.button("Copy Prompt"):
            st.write(prompt)
        if col2.button("Regenerate"):
            response = openai.Image.create(
                prompt=prompt,
                size='512x512',
                n=1
            )
            image_url = response["data"][0]["url"]
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, width=500, use_column_width=True, caption=prompt)
        if col3.button("Download Image"):
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img.save("generated_image.png")
            st.success("Image downloaded successfully!")
            
# Run the Streamlit app
if __name__ == "__main__":
    app()