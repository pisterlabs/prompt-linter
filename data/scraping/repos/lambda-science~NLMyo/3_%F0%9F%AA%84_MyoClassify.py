import streamlit as st
import joblib
import os
import numpy as np
import sys
from streamlit.components.v1 import html
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from dotenv import load_dotenv
import openai
import requests
from io import BytesIO

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
default_file_url = "https://www.lbgi.fr/~meyer/IMPatienT/sample_demo_report.pdf"


sys.path.append("../")
from src import TextReport

st.set_page_config(
    page_title="MyoClassify",
    page_icon="🪄",
)

if "id" not in st.session_state:
    st.session_state["id"] = 0


def callback():
    st.session_state["id"] += 1


@st.cache_resource()
def load_embedding_model():
    embeddings = HuggingFaceInstructEmbeddings(
        query_instruction="Represent the medicine document for classification: "
    )
    return embeddings


@st.cache_data()
def embed_text(text):
    embeddings = load_embedding_model()
    results = embeddings.embed_query(text)
    return results


@st.cache_data()
def embed_text_openai(text):
    results = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return results


@st.cache_data()
def st_analyze_pdf(uploaded_file, lang):
    pdf_object = TextReport(uploaded_file, lang=lang)
    raw_text = pdf_object.pdf_to_text()
    return raw_text


with st.sidebar:
    st.write("Report Language")
    lang = st.selectbox("Select Language", ("fra", "eng"))
    mode = st.selectbox("Select Mode", ("Instructor", "openAI"))

loaded_model_instructor_fr = joblib.load("models/instructor_fr_model_rf.joblib")
loaded_model_instructor_en = joblib.load("models/instructor_en_model_rf.joblib")
loaded_model_openai_fr = joblib.load("models/openai_fr_model_rf.joblib")
loaded_model_openai_en = joblib.load("models/openai_en_model_rf.joblib")


st.write("# MyoClassify🪄")
st.markdown(
    """
### MyoClassify🪄 a tool to automatically predict a congenital myopathy sub-type diagnosis from patient histology report PDF.
Upload a single PDF file or copy paste your text-report and the tool will automatically try to predict the congenital myopathy diagnosis among: nemaline myopathy, core myopathy, centrenoculear myopathy or non congenital myopathy (NON-MC). On the left menu you can select your repote language and the analysis mode (see bellow).  
There is two different modes to extract data from the report:  
* **Instructor** (can be less accurate but 100% private, self-hosted AI)
* **OpenAI** (more accurate but use OpenAI API, see disclaimer.)  

🚨 DISCLAIMER: If you choose OpenAI instead of private AI in tools options, some tools will use [OpenAI API](https://openai.com/). Data will be sent to OpenAI servers. If using OpenAI Model, do not upload private or non-anonymized data. As per their terms of service [OpenAI does not retain any data  (for more time than legal requirements, click for source) and do not use them for trainning.](https://openai.com/policies/api-data-usage-policies) However, we do not take any responsibility for any data leak.  

Creator and Maintainer: [**Corentin Meyer**, 3rd year PhD Student in the CSTB Team, ICube — CNRS — Unistra](https://lambda-science.github.io/)  <corentin.meyer@etu.unistra.fr>  
"""
)


st.header("PDF or Text Input")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Choose patient PDF",
        type="pdf",
        accept_multiple_files=False,
        key=st.session_state["id"],
    )
    if st.button("Load Sample PDF"):
        # Download the default file
        response = requests.get(default_file_url)
        # Convert the downloaded content into a file-like object
        uploaded_file = BytesIO(response.content)

with col2:
    input_text = st.text_area(
        "OR Write here your patient report or upload a PDF", key="input"
    )

if uploaded_file or input_text:
    if uploaded_file:
        raw_text = st_analyze_pdf(uploaded_file, lang)
        st.write("## Raw text")
        st.write(raw_text)
    else:
        raw_text = input_text
        st.write("## Raw text")
        st.write(raw_text)
    st.markdown("# Most probable diagnosis")

    if lang == "fra":
        if mode == "Instructor":
            results = embed_text(input_text)
            embedding_features = np.array(results)
            prediction = loaded_model_instructor_fr.predict(
                embedding_features.reshape(1, -1)
            )
            confidence = loaded_model_instructor_fr.predict_proba(
                embedding_features.reshape(1, -1)
            )
            label_dict = {
                i: label for i, label in enumerate(loaded_model_instructor_fr.classes_)
            }
        elif mode == "openAI":
            results = embed_text_openai(input_text)
            embedding_features = np.array(results["data"][0]["embedding"])
            prediction = loaded_model_openai_fr.predict(
                embedding_features.reshape(1, -1)
            )
            confidence = loaded_model_openai_fr.predict_proba(
                embedding_features.reshape(1, -1)
            )
            label_dict = {
                i: label for i, label in enumerate(loaded_model_openai_fr.classes_)
            }
    elif lang == "eng":
        if mode == "Instructor":
            results = embed_text(input_text)
            embedding_features = np.array(results)
            prediction = loaded_model_instructor_en.predict(
                embedding_features.reshape(1, -1)
            )
            confidence = loaded_model_instructor_en.predict_proba(
                embedding_features.reshape(1, -1)
            )
            label_dict = {
                i: label for i, label in enumerate(loaded_model_instructor_en.classes_)
            }
        elif mode == "openAI":
            results = embed_text_openai(input_text)
            embedding_features = np.array(results["data"][0]["embedding"])
            prediction = loaded_model_openai_en.predict(
                embedding_features.reshape(1, -1)
            )
            confidence = loaded_model_openai_en.predict_proba(
                embedding_features.reshape(1, -1)
            )
            label_dict = {
                i: label for i, label in enumerate(loaded_model_openai_en.classes_)
            }

    st.write("Prediction: ", prediction[0])
    st.markdown("# Probability of each diagnosis")
    for index, value in enumerate(confidence[0]):
        st.write(f"Confidence score for:  {label_dict[index]}: {round(value*100)}% ")

html(
    f"""
    <script defer data-domain="lbgi.fr/nlmyo" src="https://plausible.cmeyer.fr/js/script.js"></script>
    """
)
