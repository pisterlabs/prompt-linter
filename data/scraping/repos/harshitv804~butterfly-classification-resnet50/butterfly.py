import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
import openai
st.set_page_config(page_title="Butterfly Image Classification", page_icon="favicon.png", layout="centered", initial_sidebar_state="auto", menu_items=None)

openai.api_key= st.secrets["YOUR API KEY"]

loaded_model = load_model('butterflyresnet50.hdf5')
dataset= tf.keras.preprocessing.image_dataset_from_directory(
    'train',
    shuffle=True,
    image_size=(224,224),
    batch_size=32
    )
class_names = dataset.class_names

def prediction(image):
    img = tf.keras.preprocessing.image.load_img(image,target_size=(224, 224))
    img_array = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(img), 0)
    score = tf.nn.softmax(loaded_model.predict(img_array))
    return(class_names[np.argmax(score)].upper(),img)

def clear():
    clr_btn= st.button("Clear")
    if clr_btn:
        st.experimental_singleton.clear()

def tabs(image):
    if image != None:
        predict = st.button("Predict")
        if predict:
            butterfly, show_img = prediction(image)
            html_str = f"""
            <style>
            p.a {{
            font: bold 25px Source Sans Pro;
            }}
            </style>
            <br>
            <center>
            <p class="a">{butterfly} BUTTERFLY</p>
            </center>
            """
            st.markdown(html_str, unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 5, 3])
            col2.image(show_img, use_column_width=True)
            with st.spinner('Loading...'):
                break_line()
                result = openai.Completion.create(engine="text-davinci-003", prompt= f'give me basic biological information about {butterfly} butterfly with subtitles within hundred words', max_tokens=500)
                st.write(result.choices[0].text)
                clear()

def break_line():
    html_str = f"""
    <br>
    """
    st.markdown(html_str, unsafe_allow_html=True)

def footer_dis():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
 
#MAIN PROGRAM   
 
footer_dis()
tit1, tit2, tit3 = st.columns([1, 5, 1])
tit2.header("Butterfly Image Classification")
break_line()
col1, col2, col3 = st.columns([4, 5, 4])
col2.image("logo.gif", use_column_width=True)
break_line()


selected = option_menu(
    menu_title= None,
    options= ["Upload","Camera"],
    icons= ["image","camera"],
    menu_icon= "cast",
    default_index= 0,
    orientation= "horizontal"
    )

if selected =="Camera":
    
    cam_img = st.camera_input("Take a Picture")
    tabs(cam_img)
        
if selected =="Upload":
    
    upload_img = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    tabs(upload_img)
