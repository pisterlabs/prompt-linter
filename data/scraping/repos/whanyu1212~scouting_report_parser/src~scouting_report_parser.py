import streamlit as st
import pandas as pd
import openai
import tempfile
import plotly.express as px
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain_pydantic
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import LLMChain, StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from util.prompt_template import summary_prompt_template, qna_prompt_template
from util.attributes import ScoutingFeatures


openai.api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(
    openai_api_key=openai.api_key,
    temperature=0.7,
    model="gpt-3.5-turbo-0613",
)


def parse_upload_file(pages):
    prompt = PromptTemplate.from_template(summary_prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
    )

    summaries = stuff_chain.run(pages)
    st.subheader("Scouting Report Summary")
    st.write(summaries)


def get_player_attributes(ScoutingFeatures, pages):
    chain = create_tagging_chain_pydantic(ScoutingFeatures, llm)
    res = chain.run(pages)
    return res


def get_original_image_url(query):
    # Configure ChromeOptions for headless mode (no GUI)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    service = Service(executable_path="./chromedriver")

    # Create a Chrome WebDriver instance with the configured options
    driver = webdriver.Chrome(service=service, options=options)

    # Create a search URL using the provided query
    search_url = f"https://www.bing.com/images/search?q={query}"

    # Navigate to the search URL using the WebDriver
    driver.get(search_url)

    try:
        # Find all image elements with class name "mimg"
        image_elements = driver.find_elements(By.CLASS_NAME, "mimg")

        for image_element in image_elements:
            # Get the image URL from the element
            image_url = image_element.get_attribute("src")

            # Check if the URL is not None and doesn't end with '.gif'
            if image_url and not image_url.endswith(".gif"):
                return image_url
        else:
            return None
    except:
        return None
    finally:
        
        driver.quit()


def get_qna_answer(question, pages):
    QA_CHAIN_PROMPT = PromptTemplate.from_template(qna_prompt_template)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(pages)
    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=OpenAIEmbeddings()
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    result = qa_chain({"query": question})
    st.write(result["result"])


def radar_chart(res):
    df = pd.DataFrame(
        dict(
            r=[
                res.offense,
                res.defense,
                res.athleticism,
                res.basketball_iq,
                res.mentality,
                res.ceiling,
            ],
            theta=[
                "Offense",
                "Defense",
                "Athleticism",
                "Basketball IQ",
                "Mentality",
                "Ceiling",
            ],
        )
    )

    fig = px.line_polar(
        df,
        r="r",
        theta="theta",
        line_close=True,
        color_discrete_sequence=px.colors.sequential.Plasma_r,
        template="plotly_dark",
        range_r=[0, 10],
    )
    return fig


def main():
    st.title(":basketball: Scouting Report Parser")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

    if pdf_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.read())
            pdf_path = tmp_file.name
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            res = get_player_attributes(ScoutingFeatures, pages)

            col1, col2 = st.columns(2)  # Splitting the layout into two columns

            # Add content to the columns
            with col1:
                image_url = get_original_image_url(res.name + " " + "college")
                if image_url:
                    st.image(
                        image_url,
                        caption="Profile: " + res.name,
                        use_column_width=False,
                    )
                else:
                    st.write("No suitable high-resolution image found.")
            with col2:
                st.markdown(
                    f"""
                    <div style="margin-top: 10px; margin-left: 170px;">
                        <p>
                            Height: {res.height} <br><br><br>
                            Weight: {res.weight} <br><br><br>
                            DOB: {res.dob} <br><br><br>
                            College: {res.college} <br><br><br>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            parse_upload_file(pages)
            st.markdown("---")
            fig = radar_chart(res)
            st.subheader("Player's Radar Chart")
            st.write(fig)
            st.markdown("---")

            question = st.text_input(
                "Additional questions",
                value="Enter your question here...",
            )
            if question != "Enter your question here...":
                get_qna_answer(question, pages)


# Call the main function
if __name__ == "__main__":
    main()
