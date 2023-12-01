from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
import glob
import json
from datetime import datetime
from langchain.chains import ConversationalRetrievalChain
import time
from langchain import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from langchain.callbacks import StreamlitCallbackHandler
# from dotenv import load_dotenv
# load_dotenv()

OPENAI_API_KEY = st.secrets['openai-api-key']
pc_api_key = st.secrets['pc-api-key']
pc_env = st.secrets['pc-env']
pc_index = st.secrets['pc-index']

# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# PINECONE_API_KEY = os.environ['pc_api_key']
# PINECONE_ENVIRONMENT = os.environ['pc_env']
# index_name = os.environ['pc_index']

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

pinecone.init(
    api_key=pc_api_key,
    environment=pc_env
)

index = pinecone.Index(pc_index)

text_field = "text"

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# Site title
st.title("🤖 Learning Sciences Research Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I assist you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), max_token_limit=150, memory_key='chat_history',
                                         return_messages=True, output_key='answer')
# memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), max_token_limit=150,
#                                          memory_key='chat_history', return_messages=True, output_key='answer')


if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = memory


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def print_citations(result):
    citations = ""

    unique_citations = {}
    for doc in result['source_documents']:
        citation = doc.metadata.get('citation')
        source = doc.metadata.get('source')
        if citation:
            unique_citations[citation] = source

    for citation, source in unique_citations.items():
        citations += "- Citation: " + citation + "\n"
        citations += "  Source: " + source + "\n\n"

    return citations

def print_answer_citations_sources(result):
    output_answer = ""

    output_answer += result['answer'] + "\n\n"

    unique_citations = {}
    for doc in result['source_documents']:
        citation = doc.metadata.get('citation')
        source = doc.metadata.get('source')
        if citation:
            unique_citations[citation] = source

    for citation, source in unique_citations.items():
        output_answer += "- Citation: " + citation + "\n"
        output_answer += "  Source: " + source + "\n\n"

    return output_answer

# current_dir = os.path.dirname(os.path.abspath(__file__))
# json_file_path = os.path.join(current_dir, 'convo_history.json')
json_file_path = 'convo_history.json'

def get_convo():
    with open(json_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print("Content of convo_history.json:", content)  # Add this line for logging
    data = json.loads(content)
    return data, json_file_path


def store_convo(prompt, answers, citations):
    data, json_file_path = get_convo()
    current_dateTime = datetime.now()
    data['{}'.format(current_dateTime)] = []
    data['{}'.format(current_dateTime)].append({'Question': prompt, 'Answer': answers, 'Citations': citations})

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


prompt_template = """Use the following pieces of context to answer the question at the end. You should construct your answer only based on information provided in the context. If you don't know the answer, don't try to make up an answer, just say the exact following "Sorry, I didn't find direct answer to your question. But here are some resources that may be helpful.".

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT_revised = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3})

if prompt := st.chat_input("Ask anything about learning sciences research!"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # stream_handler = StreamHandler(st.empty())
        st_callback = StreamlitCallbackHandler(st.container())
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3})
        
        qa = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY, streaming=True, callbacks=[st_callback]), 
                                                   retriever=retriever, chain_type="stuff",
                                                   combine_docs_chain_kwargs={'prompt': QA_PROMPT_revised}, memory = memory,
                                                   verbose=True, return_source_documents=True)
        # qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, streaming=True, callbacks=[stream_handler]), 
        #                                            vectorstore.as_retriever(), 
        #                                            memory=st.session_state.buffer_memory,
        #                                            combine_docs_chain_kwargs={'prompt': QA_PROMPT_revised},
        #                                            return_source_documents=True)
        with st.spinner("searching through learning sciences research papers and preparing citations..."):
            res = qa({"question": st.session_state.messages[-1].content})
            citations = print_citations(res)
            answers = print_answer_citations_sources(res)
            store_convo(st.session_state.messages[-1].content, answers, citations)
            st.write(citations)
            # st.write(answers)  # Display the assistant's answer

    st.session_state.messages.append(ChatMessage(role="assistant", content=answers))
