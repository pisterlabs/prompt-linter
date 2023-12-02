import streamlit as st
import time
import numpy as np
import cv2
import torch
import os

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

st.markdown("<h2 style='color: green; font-style: italic; font-family: Comic Sans MS; ' >EcoVision Quest 👀🌱🌍</h2> <h3 style='color: #ADFF2F; font-style: italic; font-family: Comic Sans MS; '>Discover, Learn, Act: EcoVision for Kids!</h3>", unsafe_allow_html=True)

st.markdown("<p style='color: #4FC978; font-style: italic; font-family: Comic Sans MS; ' > Kids explore surroundings, get sustainable guidance from LLM, make eco-friendly choices, and become environmental leaders.</p>", unsafe_allow_html=True)

objects = st.text_input('Enter a list of object names separated by commas (e.g., object1, object2, object3)')
st.markdown("<p style='color: #4FC978; font-style: italic; font-family: Comic Sans MS; text-align:center; font-size:2rem' >OR</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a file")
object_list=[]

if objects:
    object_list = [obj.strip() for obj in objects.split(",")]
elif uploaded_file is not None:
    st.write("image processing started")
    st.image(uploaded_file)
    img_array = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Save the uploaded file in the "pictures" folder
    pictures_folder = 'D:/llm projects/EcoKids Hub/pictures'

    img_name = os.path.join(pictures_folder, "img.jpg")
    cv2.imwrite(img_name, img)

    object_list= obj_detect('D:/llm projects/EcoKids Hub/pictures/img.jpg')
    # for obj in object_list:
    #     st.write(obj)


title_template = PromptTemplate(
        input_variables=['object'],
        template='Let me know whether {object} is environmentally sustainable or no'
    )

title_memory = ConversationBufferMemory(input_key='object', memory_key='chat_history')

    # Llms
llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm=llm, prompt=title_template, output_key='title', memory=title_memory)

wiki = WikipediaAPIWrapper()

for obj in object_list:
    title = title_chain.run(obj)
    wiki_research = wiki.run(obj)

    st.subheader(f"Object: {obj}")
    st.write(title)
