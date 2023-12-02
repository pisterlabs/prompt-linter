import streamlit as st
from dotenv import load_dotenv
import sys
import json
from streamlit.components.v1 import html
from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp
from langchain.chat_models import ChatOpenAI
import pandas as pd
import requests
from io import BytesIO

sys.path.append("../")
from src import TextReport

st.set_page_config(
    page_title="MyoExtract",
    page_icon="📝",
)

if "id" not in st.session_state:
    st.session_state["id"] = 0

load_dotenv()


def callback():
    st.session_state["id"] += 1


with st.sidebar:
    st.write("Report Language")
    lang = st.selectbox("Select Language", ("fra", "eng"))
    mode = st.selectbox("Select Mode", ("openAI", "private-AI"))


@st.cache_resource()
def load_vicuna():
    llm_vicuna_model = LlamaCpp(
        model_path="./models/ggml-vic7b-q4_1.bin", temperature=0.01, n_ctx=2048
    )
    return llm_vicuna_model


@st.cache_data()
def json_to_dataframe(json_list):
    """
    Converts a list of JSON objects to a Pandas DataFrame.
    """
    # Create an empty DataFrame
    df = pd.DataFrame()

    # Loop through each JSON object in the list
    for json_obj in json_list:
        # Convert the JSON object to a Pandas Series
        series = pd.Series(json_obj)

        # Add the Series to the DataFrame
        df = pd.concat([df, series.to_frame().T], ignore_index=True)

    # Return the DataFrame
    return df


@st.cache_data()
def extract_first_valid_json(s):
    """
    Extracts the first valid JSON object from a string.
    """
    # Loop through each character in the string
    start_bracket = s.find("{")
    if start_bracket == -1:
        return None
    for i, c in enumerate(s[start_bracket:]):
        # Check if the current character is the start of a JSON object
        try:
            json_obj = json.loads(s[start_bracket : i + 1])
            # Return the JSON object if parsing is successful
            return json_obj
        except:
            pass

    # If no valid JSON object is found, return None
    return None


@st.cache_data()
def process_output_json(results):
    last_brace_index = results.rfind("}")
    if last_brace_index == -1:
        return results
    subset = results[: last_brace_index + 1]
    subset_extracted = extract_first_valid_json(subset)
    if subset_extracted is None:
        return results
    return subset_extracted


@st.cache_data()
def extract_metadata_openAI(input_text):
    template = """
    You are an assistant that extract informations from free text. OUTPUT MUST KEEP THE KEY NAMES AND FOLLOW THIS JSON FORMAT SIMPLY REPLACE THE VALUES. IF YOU CAN'T FIND AN INFORMATION SIMPLY INDICATE N/A DON'T TRY TO INVENT IT. Here is the list of informations to retrives, json key are indicated in parenthesis: complete name (name), age (age), birth date (birth), biopsy date (biodate), biopsy sending date (sending), muscle (muscle), biopsy number (bionumber), diagnosis (diag), presence of anomaly in PAS staining (PAS), presence of anomaly in Soudan Staining (Soudan), presence of anomaly in COX staining (COX), presence of anomaly in ATP staining (ATP), presence of anomaly in Phosrylase staining (phospho)
    Please report all dates to the format DD-MM-YYYY and all ages in years, indicate 0 if less than 1 year.
    INPUT:
    John Doe et Jane Clinton sont asymptomatique. Date de naissance: 16 février 1991, numéro de biopsie: 666-77
    Anormalie forte à la coloration PAS mais pas d'anomalie à la coloration lipide soudan. Le tableau est révêlateur d'une myopathie à némaline
    OUPUT:
    {{"name":["John Doe", "Jane Clinton"], "age":"N/A", "birth": "16-02-1991", "biodate": "N/A"", "sending": "N/A"", "muscle": "N/A"", "bionumber": "666-77", "diag": "myopathie à némaline", "PAS": "yes", "Soudan": "no", "COX": "N/A", "ATP": "N/A", "phospho": "N/A"}}
    INPUT:
    {raw_text}
    OUTPUT:

    """
    prompt = PromptTemplate(template=template, input_variables=["raw_text"])
    llm_chain = LLMChain(
        prompt=prompt, llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.01)
    )
    results = llm_chain.run(input_text)
    json_string = process_output_json(results)
    return json_string


@st.cache_data(ttl=60)
def extract_metadata_privateLLM(input_text, _llm_vicuna_model):
    template = """
    You are an assistant that extract informations from free text. OUTPUT MUST KEEP THE KEY NAMES AND FOLLOW THIS JSON FORMAT SIMPLY REPLACE THE VALUES. IF YOU CAN'T FIND AN INFORMATION SIMPLY INDICATE N/A DON'T TRY TO INVENT IT. Here is the list of informations to retrives, json key are indicated in parenthesis: complete name (name), age (age), birth date (birth), biopsy date (biodate), biopsy sending date (sending), muscle (muscle), biopsy number (bionumber), diagnosis (diag), presence of anomaly in PAS staining (PAS), presence of anomaly in Soudan Staining (Soudan), presence of anomaly in COX staining (COX), presence of anomaly in ATP staining (ATP), presence of anomaly in Phosrylase staining (phospho)
    Please report all dates to the format DD-MM-YYYY and all ages in years, indicate 0 if less than 1 year.
    INPUT:
    John Doe et Jane Clinton sont asymptomatique. Date de naissance: 16 février 1991, numéro de biopsie: 666-77
    Anormalie forte à la coloration PAS mais pas d'anomalie à la coloration lipide soudan. Le tableau est révêlateur d'une myopathie à némaline
    OUPUT:
    {{"name":["John Doe", "Jane Clinton"], "age":"N/A", "birth": "16-02-1991", "biodate": "N/A"", "sending": "N/A"", "muscle": "N/A"", "bionumber": "666-77", "diag": "myopathie à némaline", "PAS": "yes", "Soudan": "no", "COX": "N/A", "ATP": "N/A", "phospho": "N/A"}}
    INPUT:
    {raw_text}
    OUTPUT:

    """
    prompt = PromptTemplate(template=template, input_variables=["raw_text"])
    llm_chain = LLMChain(prompt=prompt, llm=_llm_vicuna_model)
    results = llm_chain.run(input_text)
    json_string = process_output_json(results)
    return json_string


@st.cache_data()
def st_analyze_pdf(uploaded_file, lang):
    pdf_object = TextReport(uploaded_file, lang=lang)
    raw_text = pdf_object.pdf_to_text()
    return raw_text


st.write("# MyoExtract 📝")
st.markdown(
    """
### MyoExtract 📝 a tool to automatically extract common metadata from patient histology report PDF to a JSON format.  
Upload a single PDF file or copy paste your text-report and the tool will automatically find for your all: complete name, age, birth date, biopsy date, biopsy sending date, muscle, biopsy number, diagnosis, presence of anomaly in PAS staining, presence of anomaly in Soudan Staining, presence of anomaly in COX staining, presence of anomaly in ATP staining,  presence of anomaly in Phosrylase staining.  

There is two different modes to extract data from the report:  
* **Private AI** (pretty slow and innacurate but 100% private, self-hosted AI)
* **OpenAI** (fast and very accurate but use OpenAI API, see disclaimer.)  

🚨 DISCLAIMER: If you choose OpenAI instead of private AI in tools options, some tools will use [OpenAI API](https://openai.com/). Data will be sent to OpenAI servers. If using OpenAI Model, do not upload private or non-anonymized data. As per their terms of service [OpenAI does not retain any data  (for more time than legal requirements, click for source) and do not use them for trainning.](https://openai.com/policies/api-data-usage-policies) However, we do not take any responsibility for any data leak.      
"""
)
default_file_url = "https://www.lbgi.fr/~meyer/IMPatienT/sample_demo_report.pdf"

st.header("PDF or Text Input")
col1, col2 = st.columns(2)
llm_vicuna_model = load_vicuna()
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
        st.write("## Raw text to analyse")
        st.write(raw_text)
    if mode == "private-AI":
        result_str = extract_metadata_privateLLM(raw_text, llm_vicuna_model)
    elif mode == "openAI":
        result_str = extract_metadata_openAI(raw_text)
    st.write("## Analysis Results (as JSON)")
    try:
        st.write(result_str)
    except Exception as e:
        st.write("Error decoding JSON raw output:")
        st.write(result_str)
        st.write(e)
    st.write("## Analysis Results (as Table)")
    try:
        table = json_to_dataframe([result_str])
        st.write(table)
    except Exception as e:
        st.write("Error with JSON, table could not be generated.")
        st.write(e)

html(
    f"""
    <script defer data-domain="lbgi.fr/nlmyo" src="https://plausible.cmeyer.fr/js/script.js"></script>
    """
)
