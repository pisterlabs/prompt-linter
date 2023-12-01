# refer this page
# https://python.langchain.com/docs/use_cases/web_scraping
import pprint

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_chat import message

load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
}


def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm, verbose=True).run(content)


@st.cache_resource
def scrape_with_playwright(urls, schema):
    print(urls)
    loader = AsyncChromiumLoader(urls)
    docs: list[Document] = loader.load()
    pprint.pprint(docs)
    bs_transfomer = BeautifulSoupTransformer()
    docs_transformed = bs_transfomer.transform_documents(docs, tags_to_extract=["span"])
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    splits: list[Document] = splitter.split_documents(docs_transformed)
    pprint.pprint(splits)
    # Process the first split
    extracted_content = extract(schema=schema, content=splits[0].page_content)
    pprint.pprint(extracted_content)
    return extracted_content


# streamlit part
st.header("Webscraping Chat")


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


with st.form(key="form", clear_on_submit=True):
    user_input: str = st.text_area("You: ", "", key="input_text", placeholder="please type target url")
    submit: bool = st.form_submit_button("Submit")


if submit:
    output: str = scrape_with_playwright(urls=[user_input], schema=schema)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.write(st.session_state["generated"][i])
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
