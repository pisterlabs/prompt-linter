from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import torch
from PIL import Image
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain


obj_template = PromptTemplate(
    input_variables = ['obj'], 
    template='Let me know if {obj} is healthy or no, in less than 10 words say why.'
)

llm = OpenAI(temperature=0.9) 

obj_chain = LLMChain(llm=llm, prompt=obj_template, output_key='health')



from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import torch
from PIL import Image
import os
import numpy as np


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def obj_detect(img):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(img)
    class_labels = results.names
    unique_class_labels = list(set(results.pred[0][:, -1].tolist()))
    detected_objects = [class_labels[int(label)] for label in unique_class_labels]
    st.image(np.squeeze(results.render()))
    return detected_objects





class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Convert the frame to RGB format (required by YOLOv5)
        frm_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

        # Perform object detection using YOLOv5
        results = model(frm_rgb)

        # Process the results
        for result in results.pred:
            for det in result:
                x, y, w, h, conf, cls = det.tolist()
                cv2.rectangle(frm, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 3)
                label = model.names[int(cls)]
                llm_res = obj_chain.run(label)
                desc = label + " " + llm_res
                cv2.putText(frm, desc, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(frm, format='bgr24')


st.markdown("<h1 style='color: #3B444B; font-style: italic; font-family: Comic Sans MS; font-size:4rem' >Healthy Wealthy NutriScan 🍎🔍 </h1> <h3 style='color:#54626F; font-style: italic; font-family: Comic Sans MS; font-size:2rem'> Real-time Healthy Food Detection. Use it while shopping or in your kitchen to figure out the healthy food and make wise choices.</h3>", unsafe_allow_html=True)

webrtc_streamer(
    key="key",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)

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
    # cv2.imwrite(img_name, img)
    st.image(img_name)

    object_list= obj_detect('D:/llm projects/EcoKids Hub/pictures/img.jpg')
    # for obj in object_list:
    #     st.write(obj)


title_template = PromptTemplate(
        input_variables=['object'],
        template='Let me know whether {object} is healthy or no'
    )

title_chain = LLMChain(llm=llm, prompt=title_template, output_key='title', )


for obj in object_list:
    title = title_chain.run(obj)

    st.subheader(f"Object: {obj}")
    st.write(title)
