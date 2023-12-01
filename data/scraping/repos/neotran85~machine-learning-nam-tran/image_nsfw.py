from openai import OpenAI
import os
import streamlit as st
import consts   
import tempfile
import PIL.Image as Image
import base64
import io

os.environ['OPENAI_API_KEY'] = consts.API_KEY_OPEN_AI
client = OpenAI()
st.title("AI-powered NSFW Image Detector")  

def get_image_base64(image_bytes):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # You can change to 'JPEG' if needed
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

# Upload the image to the Streamlit app, only PNG
upload_image = st.file_uploader("Upload an image to check if it is safe for work", type=["png", "jpg"])    
if upload_image is not None:
    with st.spinner("Checking image..."):
        image = Image.open(upload_image)
        base64_image = get_image_base64(image)
        # display the image
        try:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Check if the image is safe for work."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=100,
            )
            content = response.choices[0].message.content
            st.write(content)
            st.image(upload_image, use_column_width=True)
        except Exception as e:
            st.error("Your image is not safe for work because it contains nudity, inappropriate or offensive content.")