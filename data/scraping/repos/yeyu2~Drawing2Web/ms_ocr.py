from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
import streamlit as st
import streamlit.components.v1 as components

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
subscription_key = st.secrets["VISION_KEY"]
endpoint = st.secrets["VISION_EP"]

init_image = "https://i.ibb.co/8rngT2V/webpage.jpg"

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.ibb.co/MCWFdvb/white-1714170-1280.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

st.set_page_config(layout="wide")
add_bg_from_url() 


computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

def text_recognition(img_url):

    print("===== Read File - remote =====")
    # Get an image with text
    #read_image_url = "https://i.ibb.co/bdnFVbG/F1ua8-M6ac-AMQw-ZL.jpg"

    # Call API with URL and raw response (allows you to get the operation location)
    read_response = computervision_client.read(img_url,  raw=True)

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for it to retrieve the results 
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    # Print the detected text, line by line
    layout = []
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                #print(line.text)
                #print(line.bounding_box)
                layout.append({line.text:line.bounding_box})
    print(layout)

    print("End of Computer Vision.")
    return layout

def html_gen(layout):
    prompt = PromptTemplate(
        template="""This is a layout of a handwriting website design, 
        it including text and their coordinates of four outer vertices. 
        Make a HTML modern sans-serif website that reflect these elements and decide which 
        CSS can be used to match their relative positions, try to use proper layout tags to match
         their font size and relative placement based on their coordinates. 
         Use <ul> and <li> if the elements looks as menu list. 
         Smartly use function tags like <button> <input> if their names look as that.
         Your design should prior to the coordinates, 
         then you should also use some imagination for the layout and CSS from common web design principle.
         Remember, don't use absolute coordinates in your HTML source code. 
         Generate only source code file, no description: {layout}.\n""",
        input_variables=["layout"]
    )
    llm = ChatOpenAI(model="gpt-4-0613",temperature=0)
    chain = LLMChain(prompt=prompt, llm=llm)
    output = chain.run(layout=layout)
    print(output)

    return output

def image_run():
    html_code = ""
    layout = text_recognition(st.session_state.img)
    if layout != []:
        html_code = html_gen(layout)

    st.session_state.html = html_code
    st.session_state.image = st.session_state.img
    st.balloons()

if "html" not in st.session_state:
    st.session_state.html = ""
if "image" not in st.session_state:
    st.session_state.image = ''

st.title("Convert Your Drawing To Web Design")
col1, col2 = st.columns([0.5, 0.5], gap='medium')
with col1:
    st.text_input("Image URL:", value=init_image, key='img')
    st.button("Run", on_click=image_run)
    if st.session_state.image != '':
        st.image(st.session_state.image)
    else:
        st.image(init_image)
with col2:
    with st.expander("See source code"):
        st.code(st.session_state.html)
    with st.container():
        components.html(st.session_state.html, height=600, scrolling=True)


