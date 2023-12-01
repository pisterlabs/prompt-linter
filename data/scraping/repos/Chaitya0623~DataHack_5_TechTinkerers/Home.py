import streamlit as st
import PyPDF2
import re
import pickle
import pandas as pd
import numpy as np
from langchain.llms import OpenAI
llm = OpenAI(openai_api_key="")
from langchain import PromptTemplate
from langchain.chains import LLMChain
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import re
from googletrans import Translator
import speech_recognition as sr


string_to_check = "This is a GitHub repository."

# Use a regular expression to check if the string contains "github" (case-insensitive)


generate_example_template = """

% INSTRUCTIONS
You are a person who suggests real world applicable projects.

% TEXTUAL QUESTION
{label}

% YOUR TASK
Suggest a project that the person should do, to improve, based on their current skillset.
"""

from langchain.chains import LLMChain

with open('mnli.pkl', 'rb') as file:
    classifier = pickle.load(file)
candidate_labels =  ['Frontend (HTML, CSS, JavaScript, React, TailwindCSS, ChakraUI, ThreeJs, LaTex,Figma, Sketch, Adobe XD, Animation )',
  'Backend (NodeJS, Django, Express, MongoDB, MySQL, Go, REST_Framework)',
  'Machine Learning (Numpy, Pandas, Keras, Seaborn, Matplotlib, Scikit-Learn, Tensorflow)',
  'Computer Vision (Yolo, RCNN, Haarcascade, mediapipe, pytorch, OCR,Image Analysis, Object Detection, and Video Processing; Proficient in Deep Learning Frameworks,PyTorch, TensorFlow,CNNs)',
  'Natural Language Processing (Langchains, haystacks, Large Language Models (LLMs), Transformers)',
  'Cloud Computing (Amazon Web Services (AWS), Azure, Docker, Terraforms, Kubernetes, Google Cloud Platform, Elastic, Oracle Cloud)',
  'Blockchain (Solidity, vyper, Proof of work, Go, Rust, motoko)',
  'Programming (C, C++, Java, Python)',
  'Data Science (Google Looker Studio, PowerBi, Hadoop, R, MatLab, Julia, Scala)',
  'Graphic Designer (Branding, Print Design, Adobe, Photoshop, Illustrator, UI/UX Design)'
]
cols = ['Blockchain', 'Frontend', 'Backend', 'Machine Learning', 'Computer Vision', 'Natural Language Processing', 'Cloud Computing', 'Programming', 'Data Science', 'Graphic Designer']
dict_op = {}
key_list = []

os.environ["OPENAI_API_KEY"] = ''
embeddings = OpenAIEmbeddings()
db = FAISS.load_local('faiss_index', embeddings)

df = pd.read_csv('./hello.csv')

st.set_page_config(
    page_title="ResuGenius",
    page_icon="👋",
)

st.write("# Welcome to ResuGenius! 👋")
import streamlit as st
import pandas as pd
from io import StringIO
st.header("Upload your Resume (Any format).")

uploaded_file = st.file_uploader("Choose a file", key='1')
if uploaded_file is not None:
    st.write(uploaded_file)
    pdf = PyPDF2.PdfFileReader(uploaded_file)

    # Initialize an empty string to store the text
    pdf_text = ""

    # Loop through each page and extract text
    for page_num in range(pdf.numPages):
        page = pdf.getPage(page_num)
        page_text = page.extractText()
        pdf_text += page_text

    # Remove special characters and extra spaces
    cleaned_text = re.sub(r'[^\w\s/:-]', '', pdf_text)

    # Replace multiple spaces and line breaks with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # Strip leading and trailing spaces
    cleaned_text = cleaned_text.strip()
    output = classifier(cleaned_text, candidate_labels, multi_label=False)
    chart_data = pd.DataFrame(output['scores'], output['labels'])
    st.bar_chart(chart_data)

tab1, tab2, tab3 = st.tabs(["Fueler", "Video Resume", "MultiLingual Resume"])

with tab1:
    url_supreme = st.text_input('Add your fueler profile here ')
    if url_supreme:
        driver = webdriver.Edge()

        # # URL of the webpage to scrape
        # # url_supreme = 'https://fueler.io/cmchelsimehta'
        
        # # st.write(link)
        urls = []

        # # Open the webpage
        driver.get(url_supreme)     

        # # Find the element with the title

        try:
            elem = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/main/section/div[2]/div[2]/a')) #This is a dummy element
            )
        finally:
            title_element = driver.find_elements(By.XPATH,'/html/body/main/section/div[2]/div[2]/a')
            for ele in title_element:
                urls.append(ele.get_attribute("href"))
            driver.quit()

        
        for url in urls:
            response = requests.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                heading = soup.find("h4", class_="timeline-title").text
                # print(h4_tag.text)
                a_tag = soup.select_one('a.d-flex.align-items-center')
                href_link = a_tag['href']
                dict_op[heading] = href_link  
                div_tag = soup.find_all('div', class_='work-key-word-btn')
                key_words = {}
                key_list = []
                for i in div_tag:
                    key_list.append(i.text)
                    key_words[heading]=key_list
        st.write (dict_op)
        

    for key, value in dict_op.items():
        if re.search(r'github', value, re.IGNORECASE):
            response = requests.get(value)
            if response.status_code == 200:
                
                page_content = response.content
                soup = BeautifulSoup(page_content, 'html.parser')
                #lang_stats = soup.find('div', class_='repository-lang-stats-graph')
                span_element = soup.find_all('span', class_='color-fg-default text-bold mr-1')
                
                for i in range(len(span_element)):
                    st.write()
                    percentage_span = span_element[i].find_next('span')
                    st.write(key, span_element[i].text, {percentage_span.text})

            

with tab2:
    st.header("Add your Video Resume here:")
    # # Initialize the recognizer
    # recognizer = sr.Recognizer()

    # # Capture audio from a microphone
    # with sr.Microphone() as source:
    #     print("Speak something...")
    #     audio = recognizer.listen(source)
    #     print("Audio captured.")

    # # Use the Google Web Speech API to convert speech to text
    # try:
    #     recognized_text = recognizer.recognize_google(audio)
    #     print(f"Recognized text: {recognized_text}")
    # except sr.UnknownValueError:
    #     print("Google Web Speech API could not understand audio.")
    # except sr.RequestError as e:
    #     print(f"Could not request results from Google Web Speech API; {e}")
 

with tab3:
   st.header("Multilingual Resume")
   multi_file = st.file_uploader("Choose a file", key='2')
   if multi_file is not None:
        
        st.write(multi_file)
        pdf = PyPDF2.PdfFileReader(multi_file)

         # Create a Translator object

        # Initialize an empty string to store the text
        pdf_text = ""

        # Loop through each page and extract text
        for page_num in range(pdf.numPages):
            page = pdf.getPage(page_num)
            page_text = page.extractText()
            pdf_text += page_text
        
        translator = Translator()
        # Translate to English
        translation = translator.translate(pdf_text, dest='en')
        translated_text = translation.text

        # Remove special characters and extra spaces
        cleaned_text = re.sub(r'[^\w\s/:-]', '', translated_text)

        # Replace multiple spaces and line breaks with a single space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        # Strip leading and trailing spaces
        cleaned_text = cleaned_text.strip()
        output = classifier(cleaned_text, candidate_labels, multi_label=False)
        chart_data = pd.DataFrame(output['scores'], output['labels'])
        st.bar_chart(chart_data)


tab1, tab2, tab3 = st.tabs(["Job Search", "Team Recommender", "Project Recommender"])

with tab1:
    st.header("Semantic Search")
    st.write("Here are somejob recommendations matched with some of the job profiles!")
    st.write(dict_op.keys())
    if dict_op:
        query = (list(dict_op.keys()))[0]
        # docs = db.similarity_search(query)
        docs_and_scores = db.similarity_search_with_score(query, 2)
        st.write(docs_and_scores)
    else:
        None


with tab2:
   st.header("Team recommender ")
   option = st.selectbox(
    'What domains do you want to collaberate with?',
    ('Frontend', 'Backend', 'Machine Learning', 'Cloud Computing'))
   if option == 'Frontend':
       column_name='Backend'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)
       column_name='Machine Learning'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)
       column_name='Cloud Computing'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)
   elif option == 'Backend':
       column_name='Frontend'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)
       column_name='Machine Learning'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)
       column_name='Cloud Computing'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)
   elif option == 'Machine Learning':
       column_name='Backend'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)
       column_name='Frontend'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)
       column_name='Cloud Computing'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)
   elif option == 'Cloud Computing':
       column_name='Backend'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)
       column_name='Machine Learning'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)
       column_name='Frontend'
       max_row = df[df[column_name] == df[column_name].max()]
       st.write(max_row, column_name)

with tab3:
    st.header("Periodically recommends new Projects based on current Skill Sets.")
    label = key_list
    prompt = PromptTemplate.from_template(generate_example_template)
    formatted_prompt = prompt.format(label=label)
    chain = LLMChain(llm=llm, prompt=prompt)
    example = chain.run(formatted_prompt)
    example = example.replace('\n','')
    st.write(example)
st.sidebar.success("Select a demo above.")
