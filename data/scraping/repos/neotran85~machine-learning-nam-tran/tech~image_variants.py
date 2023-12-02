from openai import OpenAI
import os
import streamlit as st
import consts   
import tempfile
import PIL.Image as Image

os.environ['OPENAI_API_KEY'] = consts.API_KEY_OPEN_AI
client = OpenAI()
st.title("AI-powered Image Variants")  

# Upload the image to the Streamlit app, only PNG
upload_image = st.file_uploader("Upload an image", type=["png"])    
if upload_image is not None:
    with st.spinner("Generating variant..."):
        # save the image in a temp directory, PNG only
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        image = Image.open(upload_image)
        width, height = image.size
        
        # Save the image to path
        Image.open(upload_image).save(path)

        # display the image
        response = client.images.create_variation(
            image=open(path, "rb"),
            n=1,
            size="512x512"
        )
        st.image(response.data[0].url, use_column_width=True)    