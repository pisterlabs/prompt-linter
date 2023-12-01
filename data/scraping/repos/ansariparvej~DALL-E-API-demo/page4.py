import streamlit as st
import openai
from PIL import Image
from src.utils import get_width_height, resize_image
from rembg import remove

def page4():
    st.title("OpenAI DALL·E Image Editing")
    st.info("""#### NOTE: you can download image by \
    right clicking on the image and select save image as option""")

    with st.form(key='form'):
        uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg'])
        mask_file = st.file_uploader("Choose an mask file", type=['png', 'jpg'])
        prompt = st.text_input("Enter a text prompt")
        size = st.selectbox('Select size of the images', ('256x256', '512x512', '1024x1024'))
        num_images = st.selectbox('Enter number of images to be generated', (1,2,3,4))
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if (uploaded_file is not None) and (mask_file is not None) and prompt:
            our_image = Image.open(uploaded_file)
            mask_image = Image.open(mask_file)

            width, height = get_width_height(size)

            our_image = resize_image(our_image, width, height)
            mask_image = resize_image(mask_image, width, height)

            st.image(our_image, caption="Uploaded image", use_column_width=True)
            st.image(mask_image, caption="Uploaded mask", use_column_width=True)

            backround_removed_mask = remove(mask_image)

            st.image(backround_removed_mask, caption="backround_removed_mask", 
                     use_column_width=True)
            
            response = openai.Image.create_edit(
                image=our_image,
                mask=backround_removed_mask,
                prompt=prompt,
                n=num_images,
                size=size
            )

            for idx in range(num_images):
                image_url = response["data"][idx]["url"]

                st.image(image_url, caption=f"Generated image: {idx+1}",
                         use_column_width=True)


