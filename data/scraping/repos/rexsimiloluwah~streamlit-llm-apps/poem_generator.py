from typing import Optional 

import streamlit as st 
from langchain.chat_models import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate 
from langchain.chains import SequentialChain, LLMChain 

poem_title_suggestion_prompt_template = ChatPromptTemplate.from_template(
    """Suggest a title for a poem with the following topic and writing style:\n
    
    Topic: `{poem_topic}`
    Writing Style: `{poem_style}`
    """
)

poem_generator_prompt_template = ChatPromptTemplate.from_template(
    """Generate a ${poem_type} poem with the following topic and writing style:\n

    Topic: `{poem_topic}`
    Writing Style: `{poem_style}`
    """
)

@st.cache_data 
def suggest_poem_title(
    poem_topic: str, poem_style: str, openai_api_key: str, temperature: Optional[float]=0.7
):
    """Suggest a title for a poem"""
    chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature)
    poem_title_suggestion_prompt = poem_title_suggestion_prompt_template.format_messages(
        poem_topic=poem_topic,
        poem_style=poem_style
    )
    response = chat(poem_title_suggestion_prompt)
    return response.content 

@st.cache_data 
def generate_poem(
    poem_type: str, poem_topic: str, poem_style: str, openai_api_key: str, temperature: Optional[float]=0.7
):
    """Generate a poem"""
    chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature)
    poem_generator_prompt = poem_generator_prompt_template.format_messages(
        poem_type=poem_type,
        poem_style=poem_style,
        poem_topic=poem_topic
    )
    response = chat(poem_generator_prompt)
    return response.content 

