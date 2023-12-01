import streamlit as st
import openai

def page2():
    st.title("GENInteriors")
    st.info("""#### NOTE: you can download image by \
    right clicking on the image and select save image as option""")


    with st.form(key='form'):
        option1 = st.selectbox("Select room type",["Bedroom", "livingroom", "Bathroom", "HomeOffice"])
        option2 = st.selectbox("Select Style", 
                            (['Bohemian', 'Minimalist', 'Contemporary', 'Traditional']))
        option3 = st.selectbox("Select Theme", ['Tropical', 'Modern', 'Vintage', 'Zen'])
        
        prompt = f"Design a {option1} with a {option2} style and {option3} theme."
        size = st.selectbox('Select size of the images', 
                            ('256x256', '512x512', '1024x1024'))
        num_images = st.selectbox('Enter number of images to be generated', (1,2,3,4))
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if prompt:
            response = openai.Image.create(
                    prompt = prompt,
                    n = num_images,
                    size=size,
                )
            
            for idx in range(num_images):
                image_url = response["data"][idx]["url"]

                st.image(image_url, caption=f"Generated image: {idx+1}",
                         use_column_width=True)
