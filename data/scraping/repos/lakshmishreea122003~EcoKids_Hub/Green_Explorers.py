import streamlit as st
import time
import numpy as np
import cv2
import torch
import os
import random

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


def obj_detect(img):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(img)
    class_labels = results.names
    unique_class_labels = list(set(results.pred[0][:, -1].tolist()))
    detected_objects = [class_labels[int(label)] for label in unique_class_labels]
    st.image(np.squeeze(results.render()))
    return detected_objects



st.set_page_config(page_title="Image", page_icon="📈")

st.markdown("<h2 style='color: green; font-style: italic; font-family: Comic Sans MS; ' >EcoKids Green Explorers🌱🔍</h2> <h3 style='color: #ADFF2F; font-style: italic; font-family: Comic Sans MS; '>Unearth the Eco-Wonders: Discover, Recycle, and Create a Greener World! 🚀 </h3>", unsafe_allow_html=True)

st.markdown("<p style='color: #4FC978; font-style: italic; font-family: Comic Sans MS; ' > Ready to be a Green Explorer? Spot an object at home 🌿🏠, upload its pic, and win rewards 🏆! Learn how it helps the environment or can be recycled!</p>", unsafe_allow_html=True)


object_template = PromptTemplate(
    input_variables=['type'],
    template='Give one {type} object name like bicycle, plant, paper, cup, bowl, pens, bags, books, chair, or fan.'
)


llm = OpenAI(temperature=0.9)

object_chain = LLMChain(llm=llm, prompt=object_template, verbose=True, output_key='object')

obj_list = ['bicycle']
            # , 'plant', 'paper', 'cup', 'bowl', 'pens', 'bags', 'books', 'chair', 'fan']

st.markdown("<p style='color: #4FC978; font-style: italic; font-family: Comic Sans MS; ' >'Click 👇 to get the name of object' </p>", unsafe_allow_html=True)
if st.button("Get object name"):
    prompt = "eco-friendly"
    object = object_chain.run(prompt)
    st.write(':green[Find 🔍 the object given]')
    st.write(object)

st.write("💭")
st.write("💭")


check_template = PromptTemplate(
        input_variables=['object','list'],
        template='Check if {object} serves a similar purpose as any object in the list {list}. If yes, return True; otherwise, return False.'
    )

check_chain = LLMChain(llm=llm, prompt=check_template, verbose=True, output_key='check')


uploaded_file = st.file_uploader(':green[Upload the image 👇]')
object_list=[]
check = False



if uploaded_file is not None:
    st.write("image processing started")
    st.image(uploaded_file)
    img_array = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Save the uploaded file in the "pictures" folder
    pictures_folder = 'D:/llm projects/EcoKids Hub/pictures'

    img_name = os.path.join(pictures_folder, "img.jpg")
    cv2.imwrite(img_name, img)

    object_list= obj_detect('D:/llm projects/EcoKids Hub/pictures/img.jpg')
    for o in object_list:
        st.write(o)
    check = True

    check = check_chain.run(object=object, list=object_list)

info_template = PromptTemplate(
        input_variables=['object'],
        template='write for kids how {object} is either environmentally sustainable or how it can be recycled by kids.'
    )

info_chain = LLMChain(llm=llm, prompt=info_template, verbose=True, output_key='info')

if check == "True":
    st.balloons()
    st.success("Yeh you found the object 🎉🎉")
    st.write("Lets learn more 🌞")
    info = info_chain.run(object)
    st.write(info)
else:
    st.write("try again")


     




