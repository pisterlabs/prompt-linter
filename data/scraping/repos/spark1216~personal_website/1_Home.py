import streamlit as st
import requests
from streamlit_lottie import st_lottie
from streamlit_timeline import timeline
import streamlit.components.v1 as components
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from constant import *
from PIL import Image
import openai

st.set_page_config(page_title='Template' ,layout="wide",page_icon='👧🏻')

# -----------------  loading assets  ----------------- #

    
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
local_css("style/style.css")

# loading assets
lottie_gif = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_x17ybolp.json")
python_lottie = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_2znxgjyt.json")
java_lottie = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_zh6xtlj9.json")
my_sql_lottie = load_lottieurl("https://assets4.lottiefiles.com/private_files/lf30_w11f2rwn.json")
git_lottie = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_03cuemhb.json")
github_lottie = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_6HFXXE.json")
r_lottie = load_lottieurl("https://lottie.host/6154dc63-0b67-490b-9032-38b67bac2e36/mIfsJnSjZr.json")
js_lottie = load_lottieurl("https://lottie.host/fc1ad1cd-012a-4da2-8a11-0f00da670fb9/GqPujskDlr.json")



# ----------------- info ----------------- #
def gradient(color1, color2, color3, content1, content2):
    st.markdown(f'<h1 style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});font-size:60px;border-radius:2%;">'
                f'<span style="color:{color3};">{content1}</span><br>'
                f'<span style="color:white;font-size:17px;">{content2}</span></h1>', 
                unsafe_allow_html=True)

with st.container():
    col1,col2 = st.columns([8,3])

full_name = info['Full_Name']
with col1:
    gradient('#FFD4DD','#000395','e0fbfc',f"Hi, I'm {full_name}👋", info["Intro"])
    st.write("")
    st.write(info['About'])
    st.write(f'<a href="{info["LinkedIn"]}">Connect with me on LinkedIn!</a>', unsafe_allow_html=True)
    img_4 = Image.open("images/IMG_4428.png")
    st.image(img_4,width=400)
    
with col2:
    st_lottie(lottie_gif, height=280, key="data")
        

# ----------------- skillset ----------------- #
with st.container():
    st.subheader('⚒️ Skills')
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        st_lottie(python_lottie, height=70,width=70, key="python", speed=2.5)
    with col2:
        st_lottie(java_lottie, height=70,width=70, key="java", speed=4)
    with col3:
        st_lottie(my_sql_lottie,height=70,width=70, key="mysql", speed=2.5)
    with col4:
        st_lottie(git_lottie,height=70,width=70, key="git", speed=2.5)
    with col1:
        st_lottie(github_lottie,height=50,width=50, key="github", speed=2.5)
    with col2:
        st_lottie(r_lottie,height=70,width=70, key="R", speed=2.5)
    with col3:
        st_lottie(js_lottie,height=50,width=50, key="js", speed=2.5)

# -----------------  tableau  -----------------  #
with st.container():
    st.markdown("""""")
    st.subheader("📊 Airbnb Interactive Tableau Dashboard Project")
    col1,col2 = st.columns([0.95, 0.05])
    with col1:
        st.markdown(""" <a href={}> <em>🔗 access to the link </a>""".format(info['Tableau']), unsafe_allow_html=True)
    
# ----------------- Regression Project ----------------- #
with st.container():
    st.markdown("""""")
    st.subheader('✍️ US Northeast Real Estate Price Prediction Project')
    col1,col2 = st.columns([0.95, 0.05])
    with col1:            
        st.markdown(""" <a href={}> <em>🔗 access to the link </a>""".format(info['Regression Project']), unsafe_allow_html=True)
# ----------------- Time Series Project----------------- #
with st.container():
    st.markdown("""""")
    st.subheader('✍️ Sales Prediction Project using Time Series Analysis')
    col1,col2 = st.columns([0.95, 0.05])
    with col1:           
        st.markdown('<a href="https://drive.google.com/file/d/1dlDBXwfc_mP3fCE-Rb1ms3tG54xzQQh0/view?usp=sharing" target="_blank"><em>🔗 Access to the link</em></a>', unsafe_allow_html=True)

# ----------------- Experimental Design Project----------------- #
with st.container():
    st.markdown("""""")
    st.subheader('✍️ Experimental Design Project: How do song pitch, song tempo, and color of album influence the perception of the listeners?')
    col1,col2 = st.columns([0.95, 0.05])
    with col1:
        st.markdown('<a href="https://drive.google.com/file/d/1wTILf6qVTGqHR60RXAlhm9_Qtr7XNoBR/view?usp=sharing" target="_blank"><em>🔗 Access to the link</em></a>', unsafe_allow_html=True)

# ----------------- Natural Language Processing Project----------------- #
with st.container():
    st.markdown("""""")
    st.subheader('✍️ NLP Project: 2023 eBay NLP Machine Learning Competition- In Progress')
    col1,col2 = st.columns([0.95, 0.05])
    #with col1:
        #with st.expander('See the work'):
         #   components.html(embed_rss['rss'],height=400)
            
        #Update after it has been finished: st.markdown(""" <a href={}> <em>🔗 access to the link </a>""".format(info['NLP Project']), unsafe_allow_html=True)

# -----------------  contact  ----------------- #
    with col1:
        st.subheader("📨 Contact Me")
        contact_form = f"""
        <form action="https://formsubmit.co/spark400@usc.edu" method="POST">
            <input type="hidden" name="_captcha value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
        </form>
        """
        st.markdown(contact_form, unsafe_allow_html=True)
