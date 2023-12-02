from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st

from llama_index import (
    Document,
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context,
)
from llama_index.llms import OpenAI


### Streamlit Multi-Page
st.set_page_config(
    page_title="Write Details",
    page_icon="📝",
)

# Session State
if "blog_title" not in st.session_state:
    st.session_state["blog_title"] = "Blog Title"

if "outline" not in st.session_state:
    st.session_state["outline"] = "Here comes generated outline."

if "query_engine" not in st.session_state:
    st.session_state["query_engine"] = None

# if "blog_body" not in st.session_state:
#     st.session_state["blog_body"] = None

# Set the columns
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.write(f'##### {st.session_state["blog_title"]}')
    st.text_area(label="Outline", value=st.session_state["outline"], height=300)

    query_engine = st.session_state["query_engine"]
    if query_engine:
        topic = st.text_area(label="Topic for generation", height=100)
        submit_button = st.button(label="Generate Contents", type="primary")
        if submit_button:
            if not topic:
                st.error("Please input topic.")
            else:
                with st.spinner("Wait for it..."):
                    # Get response from query
                    detail_response = query_engine.query(
                        f'Write the part of the article about "{topic}" in Japanese. It is the just the part of the whole article, which means NO need the opening or finishing sentence. Return the long text as much as possible.'
                    )
    else:
        st.error("Please create query engine first.")

with col2:
    st.subheader("Generated Body")
    if submit_button:
        st.text_area(label="Blog Body", value=detail_response.response, height=500)
        # Save as a file
        st.download_button(
            label="Download the body as a text file",
            data=detail_response.response,
            file_name=f'{st.session_state["blog_title"]}_body.txt',
            mime="text/plain",
        )


# def save_blog_body():
#     st.session_state["blog_body"] = blog_body


# with col3:
#     st.subheader("Blog Body")
#     blog_body = st.text_area(
#         label="Blog Body",
#         value=st.session_state["blog_body"],
#         height=500,
#         on_change=save_blog_body,
#     )
