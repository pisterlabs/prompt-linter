import streamlit as st
import openai
import os
import requests

### CONFIG
# set image directory
image_dir = os.path.join(os.curdir,'images')

# create the directory if it doesn't yet exist
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
# clear any existing files down
if len(os.listdir('images')) > 0:
    os.system("rm images/*")

# set URL directory
url_dir = os.path.join(os.curdir,'urls')
# create the directory if it doesn't yet exist
if not os.path.isdir(url_dir):
    os.mkdir(url_dir)
# clear any existing files down
if len(os.listdir('urls')) > 0:
    os.system("rm urls/*")


# set API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

### DALLE PRINT APP

st.title('DALL-E Print')

st.subheader('Dazzle us with your creativity')

prompt = st.text_input(
    "Enter a prompt for the image you'd like to generate below",
    "",
    key="promptSubmission",
)

num_images = st.number_input('Number of images to generate',min_value=3,max_value=3)

size_option = option = st.selectbox(
    'What size of images would you like?',
    ('256x256', '512x512', '1024x1024'))

if prompt != "":
    st.write(f'You want to generate {prompt} {num_images} times at a size of {size_option}')

if st.button('Submit', key='generationSubmit'):
    with st.spinner('Casting magic...'):
        generation_response = openai.Image.create(
        prompt=prompt,
        n=int(num_images),
        size=size_option,
        response_format="url",
    )

    
    with st.spinner('Saving files...'):
        counter = 0
        for image in generation_response['data']:
            
            counter += 1

            generated_image_name = f"generated_image_{counter}.png"  # any name you like; the filetype should be .png
            generated_image_filepath = os.path.join(os.getcwd(),'images', generated_image_name)
            generated_image_url = image["url"]  # extract image URL from response
            generated_image = requests.get(generated_image_url).content  # download the image
            url_path = os.path.join(os.curdir,'urls',f'url_{counter}.txt')

            with open(generated_image_filepath, "wb") as image_file:
                image_file.write(generated_image)  # write the image to a file

            with open(url_path, "w") as url_file:
                url_file.write(image['url'])  # write the url to a file
            

    st.write("Files saved, time to pick one!")

else:
    #st.button('Submit')
    st.write('Click above to submit your design')
    #print(prompt)


    