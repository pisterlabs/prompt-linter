import logging

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from core.util import news_post_download
from db import Post, publish

st.set_page_config(page_title="Translate Telugu News")
st.sidebar.header("News Translator")
my_form = st.form(key="form")
input = my_form.text_input(
    "URL",
    key="url",
    placeholder="https://www.eenadu.net/telugu-news/india/sudha-murty-files-complaint-over-misuse-of-her-name-for-event-promotion-in-us/0700/123177262",
)
submit = my_form.form_submit_button(label="Submit")


def get_data():
    # Make an asynchronous call to an API or database
    result = news_post_download(input)
    chat = ChatOpenAI(temperature=0, model_name="gpt-4")
    response_schemas = [
        ResponseSchema(
            name="post",
            description="Rewrite translated text in left wing style",
        ),
        ResponseSchema(
            name="huff_title",
            description="Suggest Huffington Post style headline for the article",
        ),
        ResponseSchema(
            name="nypost_title",
            description="Suggest Newyork Post style headline for the article",
        ),
        ResponseSchema(
            name="daily_caller",
            description="Suggest Daily Caller style headline for the article",
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    template = """
       Your reporter assisting to write news article complete below tasks
       1. Translate text in backticks to english
       {format_instructions}
       ```{docs}```
       """
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate(
        messages=[HumanMessagePromptTemplate.from_template(template)],
        input_variables=["docs"],
        partial_variables={"format_instructions": format_instructions},
    )

    _input = prompt.format_prompt(docs=result)
    logging.info(f"raw post: {result}")
    output = chat(_input.to_messages())
    logging.info(f"output: {output.content}")
    result = output_parser.parse(output.content)
    # Return the data
    return result


if submit:
    result = get_data()
    publish(Post(
        huff_title=result["huff_title"],
        nypost_title=result["nypost_title"],
        daily_caller=result["daily_caller"],
        url=input,
        post=result["post"]
    ))
    st.header(result["huff_title"])
    st.header(result["nypost_title"])
    st.header(result["daily_caller"])
    st.write(result["post"])
