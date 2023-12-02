from pickle import bytes_types
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import pytesseract

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


def gptchain(inputs,transcript):
    # define the output
    response_schemas = [
        ResponseSchema(name="sentiment", description="a sentiment label based on the past content. It should be either Negative or Positive"),
        ResponseSchema(name="idea", description="""
        If the sentiment is Negative then return the reason why the user shouldn't be interested in this content, along with its danger.
        If the sentiment is Positive then return the encouragement to make the user interest in this content even more, along with other relevant content.
        For Neutral then return a instruct for a better content. 
        """),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # prompt template
    template = """You are good at detecting harmful, content and also encouraging good and useful one. All labels that you know are Negative and Positive.
    Given a past content, your job is to answer as best as possible.
    The past content:
    {chat_history}

    Instructions:
    {format_instructions}
    The question: {question}."""

    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(template=template, input_variables=["chat_history", "question"],
                            partial_variables={"format_instructions": format_instructions})

    memory = ConversationBufferMemory(memory_key="chat_history")
    memory.chat_memory.add_user_message("This is what I have read recently")
    memory.chat_memory.add_user_message(transcript)
    # Build chain
    model = OpenAI(verbose=True, temperature=0.0)
    sentiment_chain = LLMChain(llm=model, prompt=prompt, verbose=True, memory=memory, output_key='result')

    ans = sentiment_chain({"question": inputs})

    return ans['result']
        
        
        
def uploadfile():
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])    
    print("Uploaded File :", uploaded_file)
    if uploaded_file is not None:
        #content = Image.open(uploaded_file)
        data = pytesseract.image_to_string(Image.open(uploaded_file))
        st.write(data)
        st.write(gptchain("What do you think about the past content",data))
    
def renderimagePage():
    st.title("Sentiment Analysis your image")
    components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 10px" /> """)
    # st.markdown("### User Input Text Analysis")
    st.subheader("Image Analysis")
    st.text("Input an image you want:") 
    uploadfile()
    
    
    
