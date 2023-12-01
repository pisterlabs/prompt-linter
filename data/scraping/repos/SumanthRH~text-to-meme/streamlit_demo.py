# imports
import sys
# sys.path.append('.')
from utils.sbert_meme_classifier import Classifier
import pickle
from PIL import Image
import os
import matplotlib.pyplot as plt
from gpt3_demo import PrimeGPT
import openai
from PIL import Image, ImageOps, ImageFont, ImageDraw
from utils.draw_utils import draw_caption_and_display
import streamlit as st
import base64


datapath = "data/gpt3_user_prompt_dic.pkl"

# constants
gpt3_engine = 'text-davinci-002'
temperature=0.7
max_tokens=256
frequency_penalty=0.0
presence_penalty=0.0
DATA_PATH = "data/"


footer="""<style>
a:link , a:visited{
color: #00BFF;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #121212;
color: #FFFFFF;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a href="https://sumanthrh.com/" target="_blank">Sumanth Hegde</a>, <a href="https://yashsk.com/" target="_blank">Yash Khandelwal</a> and <a href="https://www.linkedin.com/in/wonsuk-jang-3b33a819a/" target="_blank">Wonsuk Jang</a></br>
<a href=https://github.com/SumanthRH/text-to-meme#terms-of-use>Terms of Use</a></p>
</div>
"""


LOGO_IMAGE = "logo.png"

st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
        color: #f9a01b !important;
    }
    .logo-img {
        float:right;
        width: 70px;
        height: 70px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# padding-top: 75px !important;
st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        <p class="logo-text"> &nbsp Jester</p>
    </div>
    """,
    unsafe_allow_html=True
)
# st.components.v1.html(html)

@st.cache_resource
def get_data():
    with open("data/meme_900k_cleaned_data_v2.pkl", 'rb') as f:
        data = pickle.load(f)
    return data
@st.cache_resource
def get_clf():
    model_name = 'sentence_transformer_roberta_samples_100_epochs_5'
    # model_name = 'roberta_base'
    clf = Classifier(model_name=model_name, k=15)
    return clf

def show_images():
    ind = st.session_state.image_ind
    file_name = st.session_state.paths[ind - 1]
    img = Image.open(os.path.join(DATA_PATH, file_name))
    img = img.convert(mode="RGB")
    st.session_state.img = img

def get_templates():
    # initialize classifier
    predictions = st.session_state.clf.predictTopK(text=st.session_state.prompt)
    paths = [st.session_state.data['uuid_image_path_dic'][uuid] for uuid in predictions]
    st.session_state.paths = paths
    st.session_state.uuids = predictions
    st.session_state.labels = [st.session_state.data['uuid_label_dic'][uuid] for uuid in predictions]
    # st.session_state.image_ind = 1
    show_images()
    # return paths, predictions

# st.title("Jester")
st.write("Welcome to Jester, a text-to-meme generation engine. Enter any prompt and you will get 10 relevant meme templates to select from. Pick a template, and click on 'Generate Meme' to get a meme. We use GPT-3 to generate captions, so please enter your API key below!")

if st.button("Click on Me for some examples!"):
    st.markdown("Alright, here are some example prompts you can use:</br>"
                "1. *Why is the commercial not the same volume as the show ugghh*</br>"
                "2. *I forgot to turn the lights out before my trip!*</br>"
                "3. *When you give me the gummy bears, you can leave*</br>"
                "4. *This is Spartaa*", unsafe_allow_html=True)

# Setup radio buttons
radio_vals = ["With GPT-3", "Without GPT-3"]
gen_val = st.radio("Jester Mode", radio_vals)

if gen_val == radio_vals[0]:
    api_key = st.text_input("OpenAI API Key", type="password", key='api_key')
    if len(api_key):
        if "gpt" not in st.session_state:
            gpt = PrimeGPT(st.session_state.api_key, datapath, gpt3_engine, temperature, max_tokens)
            st.session_state['gpt'] = gpt

        if "data" not in st.session_state:
            st.session_state['data'] = get_data()

        if "clf" not in st.session_state:
            st.session_state['clf'] = get_clf()

            # "Why is the commercial not the same volume as the show uggh"
        prompt = st.text_input("Enter any text below 👇", on_change=get_templates, key="prompt")
        image_ind = st.slider(label="Select your favourite template!", min_value=1, max_value=10, step=1,
                              on_change=show_images,
                              key="image_ind")
        if len(prompt):
            if "paths" not in st.session_state:
                get_templates()

            ind = st.session_state.image_ind
            st.image(st.session_state.img, caption=st.session_state.labels[ind - 1].replace('-', " "))
            # st.image(st.session_state.img, caption=st.session_state.uuids[ind-1])

            if st.button("Generate Meme"):
                file_name = st.session_state.paths[ind - 1]
                img = Image.open(os.path.join(DATA_PATH, file_name))
                img = img.convert(mode="RGB")
                uuid = st.session_state.uuids[ind - 1]
                st.session_state.gpt.prime_gpt_from_uuid(uuid)
                gpt_prompt = st.session_state.gpt.gpt.get_prime_text()
                label = st.session_state.data['uuid_label_dic'][uuid].replace("-", " ")
                prompt_begin = f"Give a humourous, witty meme caption based on the input provided. The label of this meme is '{label}'\n\n"
                gpt_prompt = prompt_begin + gpt_prompt + "input:" + prompt + "\noutput:"
                with open("gpt_prompt.txt", 'w') as f:
                    f.write(gpt_prompt)
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=gpt_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )

                img_caption = None
                img = draw_caption_and_display(img, response, return_img=True)
                st.image(img, caption=img_caption)
else:
    st.write("Ok, so you won't get witty captions from GPT-3")

    if "data" not in st.session_state:
        st.session_state['data'] = get_data()

    if "clf" not in st.session_state:
        st.session_state['clf'] = get_clf()

    prompt = st.text_input("Enter any text below 👇", on_change=get_templates, key="prompt")
    image_ind = st.slider(label="Select your favourite template!", min_value=1, max_value=10, step=1, on_change=show_images,
                          key="image_ind")
    
    if len(prompt):
        if "paths" not in st.session_state:
            get_templates()

        ind = st.session_state.image_ind
        st.image(st.session_state.img, caption=st.session_state.labels[ind-1].replace('-', " "))
        # st.image(st.session_state.img, caption=st.session_state.uuids[ind-1])
        file_name = st.session_state.paths[ind - 1]
        img = Image.open(os.path.join(DATA_PATH, file_name))
        
        img_caption = None
        
        if st.button("Generate Meme"):
            img = img.convert(mode="RGB")
            img = draw_caption_and_display(img, response=prompt, return_img=True)
            st.image(img, caption=img_caption)


st.markdown(footer,unsafe_allow_html=True)
