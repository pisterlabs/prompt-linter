import streamlit as st
import pandas as pd
import numpy as np
import openai 
from dotenv import load_dotenv
import os
#new section
import os
import streamlit as st
from PyPDF2 import PdfReader
import langchain
langchain.verbose = False
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
  
#new section
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
def get_reply(text, messages):
    # Define the initial message containing system instructions and previous messages
    prompt = f"""
            You will be provided with a query related law or crime or in general the rights of a person. You are an AI bot named "VIDHAN" developed to help people of india according to your knowledge of "INDIAN LAW AND CONSITUTION" 
            <>.
            check the query 
            If the query is not related to law or legal help, then simply write \"Please ask a legal query\"
            If the query is related to Law or Legal help , Perform the following actions.
            1. Analyze the problem completely.
            2. Think about possible solutions which are according to the Indian law only.
            3. Use only the Indian law and constitution to answer.
            4. Answer in small points and be precise and helpful.
            5. Provide  personalized answer.

           

         

            """
    message = [
        {"role": "system", "content": f"""understand if a query is related to law or legal help """}, 
        {"role": "system", "content": prompt}, 
        {"role": "user", "content": "i have become a victim to a theft and all of my money is being stolen what should i do"},
        {"role": "assistant", "content": "you should visit the nearest police station and report the stolen money to the police and under the section of theft and law, the police will take action and report back to you "},
        {"role": "user", "content": "Who is Shahrukh khan"},
        {"role": "assistant", "content": "Please ask a legal query"}, 
    ]

    # Create an empty reply
    reply = ""

    # Extend the initial message with previous messages from the chat history
    message.extend([{"role": m["role"], "content": m["content"]} for m in messages])

    # Use the OpenAI API to stream responses
    for response in openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=message,
        stream=True,
        temperature=0.8,
        max_tokens=400
    ): 
        reply += response.choices[0].delta.get("content", "")

    return reply
st.title("VIDHAN :  THE LEGAL BOT ")
# Create a radio button to choose the chat mode
chat_mode = st.radio("Select Chat Mode:", ("Legal Document Review", "LegalBot"))
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if chat_mode == "Legal Document Review":
    # PDF Chat Logic
    pdf = st.file_uploader("Upload your PDF for review", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        max_length = 1800
        original_string = text
        temp_string = ""
        strings_list = []

        for character in original_string:
            if len(temp_string) < max_length:
                temp_string += character
            else:
                strings_list.append(temp_string)
                temp_string = ""

        if temp_string:
            strings_list.append(temp_string)

        # Split into chunks

        # Create embeddings (You may need to replace this with your specific code)
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(strings_list, embedding=embeddings)

    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    user_question = prompt

    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.9)

        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

elif chat_mode == "LegalBot":
    # Legal Bot Logic
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    if prompt:
        ans = get_reply(prompt, st.session_state.messages)
        response = f"Bot: {ans}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})



