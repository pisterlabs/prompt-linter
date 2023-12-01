import streamlit as st 
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader, QuestionAnswerPrompt, LLMPredictor, ServiceContext
import json
from langchain import OpenAI
from llama_index import download_loader
from tempfile import NamedTemporaryFile
from llama_index import (
    GPTVectorStoreIndex,
    ResponseSynthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine




def process_pdf(uploaded_file):
    loader = PDFReader()
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        documents = loader.load_data(file=Path(temp_file.name))
    
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=1900))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    
    if "index" not in st.session_state:
        index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
        retriever = index.as_retriever(retriever_mode='embedding')
        index = RetrieverQueryEngine(retriever)
        st.session_state.index = index
    # st.session_state.index = index
    return st.session_state.index
        


def call_openai(source):
    messages=[{"role": "user", "content": source}]

    response = openai.ChatCompletion.create(
        model="gpt-4-0314",
        max_tokens=7000,
        temperature=0.1,
        messages = messages
       
    )
    return response.choices[0].message.content

st.title("CourseBot")
st.caption("AI-powered course creation made easy")
DATA_DIR = "data"
PDFReader = download_loader("PDFReader")
loader = PDFReader()



######################       defining tabs      ##########################################




######################       Upload chapter column      ##########################################

uploaded_file = st.file_uploader("Upload a Chapter as a PDF file", type="pdf")
toc_option = st.radio("Choose a method to provide TOC", ("Generate TOC", "Copy Paste TOC"))
forma = """"{
  "Topics": [
    {
      "Topic 1": [
        "Subtopic 1.1",
        "Subtopic 1.2",
        "Subtopic 1.3"
      ]
    },
    {
      "Topic 2": [
        "Subtopic 2.1",
        "Subtopic 2.2",
        "Subtopic 2.3"
      ]
    },
     continue with topics...
  ]
}

"""
if uploaded_file is not None:
     
        index = process_pdf(uploaded_file)
        if "index" not in st.session_state:
            st.session_state.index = index

        st.success("Index created successfully")
     

if toc_option == "Generate TOC":
    toc = st.button("Genererate TOC")
    try:
        if toc:
            toc_res = st.session_state.index.query(f" create a table of contents with topics and subtopics by reading through the document and create a table of contents that accurately reflects the main topics and subtopics covered in the document. The table of contents should be in the following format: " + str(forma))
            str_toc = str(toc_res)
            table_of_contents = json.loads(str_toc)

            if "table_of_contents" not in st.session_state:
                st.session_state.table_of_contents = table_of_contents
            st.write(st.session_state.table_of_contents)

            st.success("TOC loaded, Go to the next tab")

    except (KeyError, AttributeError) as e:
        print("Error generating TOC")
        print(f"Error: {type(e).__name__} - {e}")


elif toc_option == "Copy Paste TOC":
    toc_input = st.text_area("Paste your Table of contents:")

    if st.button("Save TOC"):
        try:
            # table_of_contents = json.loads(toc_input)
            src =  "Convert the following table of contents into a json string, use the JSON format given bellow:\n"+ "Table of contents:\n"+ toc_input.strip() + "\n JSON format:\n"+ str(forma) + ". Output should be a valid JSON string."
            toc_res = call_openai(src)
            str_toc = str(toc_res)
            table_of_contents = json.loads(str_toc)
            # st.write(table_of_contents)

            if "table_of_contents" not in st.session_state:
                st.session_state.table_of_contents = table_of_contents
            st.write(st.session_state.table_of_contents)

        except json.JSONDecodeError as e:
            st.error("Invalid JSON format. Please check your input.")





######################       refining toc start      ##########################################
