import streamlit as st

import pinecone
import tiktoken
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


tokenizer = tiktoken.get_encoding('cl100k_base')
model_name = 'text-embedding-ada-002'

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

embed = OpenAIEmbeddings(
    #model=model_name,
    openai_api_key=st.secrets["openai_key"],
)
pinecone.init(
    api_key=st.secrets["pinecone_key"],
    environment=st.secrets["pinecone_env"]
)
index = pinecone.Index(st.secrets["pinecone_index"])





st.title("OTSEC GPT")
st.image('otsecgpt.png', width=400)
st.write("I am an Industrial Control Systems cybersecurity expert, ask me anything :)")
user_input = st.text_input("Enter your question:")
if st.button("Ask!"):
        st.markdown("### Sources ###")
        with st.spinner('Searching similar data'):
            try:
                text_field = "text"
                vectorstore = Pinecone(
                    index, embed.embed_query, text_field
                )
                context = vectorstore.similarity_search(
                user_input,  # our search query
                k=3  # return 3 most relevant docs
                )
                st.success('Top 3 most similar content identifed!')
                for answer in context:
                    if('url' in answer.metadata):
                        expander = st.expander('📺' + answer.metadata['title'])
                    elif('type' in answer.metadata and answer.metadata['type'] == 'book'):
                        expander = st.expander('📘' + answer.metadata['title'])
                    else:
                        expander = st.expander('🧑‍🔧' + answer.metadata['title'])
                    if('url' in answer.metadata):
                            expander.write('📺 ' + answer.metadata['url'])
                    expander.write(answer.page_content)
                    retrieval_error = False
            except:
                retrieval_error  = True
                st.warning('Retrieval failed. Please click "Ask" again')

        if not retrieval_error:
            messages = [
                SystemMessage(
                    content="You are a helpful assistant that explains cybersecurity. All your answers must suggest the most secure way to do things. All the questions you receive are in the context of Industrial Control Systems."
                ),
                HumanMessage(
                    content=user_input
                )
            ]
            messages_with_context = [
                SystemMessage(
                    content="You are a helpful assistant that explains cybersecurity. All your answers must suggest the most secure way to do things. All the questions you receive are in the context of Industrial Control Systems."
                ),
                SystemMessage(
                    content="You can answer based on the following trusted information : " + str(context)
                ),
                HumanMessage(
                    content=user_input
                )
            ]

        if not retrieval_error:
            st.markdown("### Open AI results ###")
            with st.spinner('Sending the request to OpenAI'):
                # GPT3.5
                gpt35 = OpenAI(openai_api_key = st.secrets["openai_key"], model_name="text-davinci-003")
                gpt35_answer = gpt35(user_input)
                # GPT4
                chat = ChatOpenAI(temperature=0, openai_api_key=st.secrets["openai_key"], model_name="gpt-4")
                answer = chat(messages)
                #GPT4 with embeddings
                chat_with_context = ChatOpenAI(temperature=0, openai_api_key=st.secrets["openai_key"], model_name="gpt-4")
                answer_with_context = chat(messages_with_context)
            st.success("Answers received from OpenAI")
            tab1, tab2, tab3 = st.tabs(["Davinci003", "GPT-4", "GPT-4 with context"])
            box_color = "azure"
            with tab1:
                st.header("Davinci003")
                st.markdown(f'<div style="background-color: {box_color}; padding: 10px;">'
                f'<p style="color: black;">{gpt35_answer}</p>'
                f'</div>', unsafe_allow_html=True)
            with tab2:
                st.header("GPT-4")
                st.markdown(f'<div style="background-color: {box_color}; padding: 10px;">'
                f'<p style="color: black;">{answer.content}</p>'
                f'</div>', unsafe_allow_html=True)
            with tab3:
                st.header("GPT-4 with context")
                st.markdown(f'<div style="background-color: {box_color}; padding: 10px;">'
                f'<p style="color: black;">{answer_with_context.content}</p>'
                f'</div>', unsafe_allow_html=True)
        

        
