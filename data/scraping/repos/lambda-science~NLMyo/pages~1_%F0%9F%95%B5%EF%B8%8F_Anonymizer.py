import os
import streamlit as st
from streamlit.components.v1 import html
from zipfile import ZipFile
import uuid
from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp
from langchain.chat_models import ChatOpenAI
import requests
from io import BytesIO
import json
import sys

sys.path.append("../")
from src import TextReport

st.set_page_config(
    page_title="Anonymizer",
    page_icon="🕵️",
)


if "id" not in st.session_state:
    st.session_state["id"] = 0


def callback():
    st.session_state["id"] += 1


class NamedBytesIO(BytesIO):
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop("name", "default_name")
        BytesIO.__init__(self, *args, **kwargs)


@st.cache_resource(ttl=1)
def st_analyze_pdf(uploaded_file, lang, mode="regex"):
    pdf_object = TextReport(uploaded_file, lang=lang)
    if mode == "openAI":
        raw_text = pdf_object.pdf_to_text()
        result_dict = censor_openAI(raw_text)
        to_censor_elem = [item for sublist in result_dict.values() for item in sublist]
        path_pdf = pdf_object.pdf_censor(
            "results", mode=mode, to_censor_list=to_censor_elem
        )
        censor_file = open(path_pdf, "rb")
    elif mode == "regex":
        path_pdf = pdf_object.pdf_censor("results", mode=mode)
        censor_file = open(path_pdf, "rb")
    return path_pdf, censor_file


@st.cache_resource(ttl=3600)
def st_zip_file(uploaded_file, lang, mode="regex"):
    zip_name = str(uuid.uuid4()) + ".zip"
    with ZipFile(zip_name, "w") as zipObj:
        for file in uploaded_file:
            path_pdf, censor_file = st_analyze_pdf(file, lang, mode)
            zipObj.writestr(path_pdf.split("/")[-1], censor_file.read())
            try:
                os.remove(path_pdf)
            except:
                pass
    zip_file = open(zip_name, "rb")
    return zip_name, zip_file


@st.cache_resource()
def load_vicuna():
    llm_vicuna_model = LlamaCpp(
        model_path="./models/ggml-vic7b-q4_1.bin", temperature=0.01, n_ctx=2048
    )
    return llm_vicuna_model


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
def censor_openAI(input_text):
    template = """
    You are an assistant that extract names and dates from free text. OUTPUT MUST KEEP THE KEY NAMES AND FOLLOW THIS JSON FORMAT SIMPLY REPLACE THE VALUES. IF YOU CAN'T FIND AN INFORMATION SIMPLY INDICATE N/A DON'T TRY TO INVENT IT. Extract all names you can find as a list of names under the "names" key and all dates you can find as a list of raw dates in the key "dates".
    INPUT:
    Je confirme avoir bien reçu le 2 mai 2023 les biopsies de Jérémy Legrand et de Clara nés respectivement le 09/06/1997 et 08 12 2003, tous deux atteins de myopathie congénitale. Signé Dr. Jane Doe -
    OUPUT:
    {{"names":["Jérémy Legrand", "Clara", "Jane Doe"], "dates":["2 mai 2023", "09/06/1997", "08 12 2003"]}}
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


st.write("# Anonymizer🕵️")
st.markdown(
    """
### Anonymizer🕵️ a tool to automatically censor patient histology report PDF.  
Upload multiple PDF files and the tool will automatically censor the patient name, date of birth, date of the report and give you a download link to the archive.  
No data are stored on the server, everything is erased right after the anonymization.  
There is three different modes to anonymize the reports: 
* **Regex** (fast but innacurate, does not rely on AI)
* **OpenAI** (fast and very accurate but use OpenAI API, see disclaimer.)  

🚨 DISCLAIMER: If you choose OpenAI on the left, this tool will use [OpenAI API](https://openai.com/). Data will be sent to OpenAI servers. If using OpenAI Model, do not upload private or non-anonymized data. As per their terms of service [OpenAI does not retain any data  (for more time than legal requirements, click for source) and do not use them for trainning.](https://openai.com/policies/api-data-usage-policies) However, we do not take any responsibility for any data leak.    

Creator and Maintainer: [**Corentin Meyer**, 3rd year PhD Student in the CSTB Team, ICube — CNRS — Unistra](https://lambda-science.github.io/)  <corentin.meyer@etu.unistra.fr>  
"""
)
st.markdown("**Upload multiple PDF**")
st.info("10 PDFs takes ~1min to be processed.")

with st.sidebar:
    st.write("Report Language")
    lang = st.selectbox("Select Language", ("fra", "eng"))
    mode = st.selectbox("Select Mode", ("regex", "openAI"))


default_file_url = "https://www.lbgi.fr/~meyer/IMPatienT/sample_demo_report.pdf"


st.write("Upload your list of PDF OR click the Load Sample PDF button")
uploaded_file = st.file_uploader(
    "Choose PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key=st.session_state["id"],
)
if st.button("Load Sample PDF"):
    # Download the default file
    response = requests.get(default_file_url)
    # Convert the downloaded content into a file-like object
    file_object = NamedBytesIO(response.content, name="sample.pdf")
    file_path = os.path.join(file_object.name)
    with open(file_path, "wb") as file:
        file.write(response.content)
    uploaded_file = [file_object]


if uploaded_file != []:
    st.write(f"{len(uploaded_file)} file(s) uploaded !")
    # random UUID name for zip file

    zip_name, zip_file = st_zip_file(uploaded_file, lang, mode=mode)
    st.success(
        "All reports have been processed ! You can now download the archive. Click on the donwload button below will reset the page."
    )
    st.download_button(
        "Download Censored Report",
        zip_file,
        file_name=zip_name,
        key="download",
        on_click=callback,
    )
    try:
        os.remove(zip_name)
    except:
        pass

html(
    f"""
    <script defer data-domain="lbgi.fr/nlmyo" src="https://plausible.cmeyer.fr/js/script.js"></script>
    """
)
