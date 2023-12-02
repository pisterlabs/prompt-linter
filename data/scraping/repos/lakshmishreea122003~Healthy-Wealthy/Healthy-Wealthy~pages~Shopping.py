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


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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



st.markdown("<h1 style='color: #3B444B; font-style: italic; font-family: Comic Sans MS; font-size:4rem' >Healthy Wealthy HealthScan </h1> <h3 style='color:#54626F; font-style: italic; font-family: Comic Sans MS; font-size:2rem'> Real-time Healthy Food Detection. Use it while shopping or in your kitchen to figure out the healthy food and make wise choices.</h3>", unsafe_allow_html=True)

webrtc_streamer(
    key="key",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)