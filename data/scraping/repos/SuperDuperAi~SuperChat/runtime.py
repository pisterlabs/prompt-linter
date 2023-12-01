import time
import boto3
import streamlit as st
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from botocore import config

with open("styles.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    config=config.Config(
        connect_timeout=1000,
        read_timeout=3000
    )
)


# add to sidebar inputs max_tokens_to_sample
st.sidebar.subheader('Model parameters')
max_tokens_to_sample = st.sidebar.slider('tokens to answer', 256, 8000, 4000)


llm = Bedrock(client=bedrock_runtime, model_id="anthropic.claude-v2")
llm.model_kwargs = {"temperature": 0.7, "max_tokens_to_sample": max_tokens_to_sample}


@st.cache_resource
def load_llm():
    DEFAULT_TEMPLATE = """{history}\n\nHuman: {input}\n\nAssistant:"""
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=DEFAULT_TEMPLATE
    )

    model = ConversationChain(
        prompt=prompt,
        llm=llm,
        # verbose=True,
        memory=ConversationBufferMemory(
            human_prefix="\n\nHuman: ",
            ai_prefix="\n\nAssistant:"
        )
    )

    return model


model = load_llm()
