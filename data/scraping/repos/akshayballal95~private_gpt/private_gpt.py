from langchain import ConversationChain,PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import  VectorstoreIndexCreator
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.llms import GPT4All
from langchain.memory import VectorStoreRetrieverMemory

import streamlit as st
from streamlit_chat import message

import torch 



loader = DirectoryLoader('D:/OneDrive/Documents/Obsidian/Projects/myVault/', glob="**/*.md",recursive=True, show_progress=True, use_multithreading=True, loader_cls=TextLoader)
docs = loader.load()
len(docs)


embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
llm = GPT4All(model="./ggml-mpt-7b-instruct.bin", top_p=0.5, top_k=0,  temp=0.5, repeat_penalty=1.1, n_threads=12, n_batch=8, n_ctx=2048)


index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])
retriever = index.vectorstore.as_retriever(search_kwargs=dict(k=2))
memory = VectorStoreRetrieverMemory(retriever=retriever)


_DEFAULT_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.
###Instruction: The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Do not make up answers and provide only information that you have.
Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)
{input}

### Response:
"""

PROMPT = PromptTemplate(
    input_variables=[ "history", "input"], template=_DEFAULT_TEMPLATE
)

conversation_with_summary = ConversationChain(
    llm=llm, 
    prompt=PROMPT,
    # We set a very low max_token_limit for the purposes of testing.
    memory = memory,
    verbose=True
    )



st.set_page_config(
    page_title="PrivateGPT",
    page_icon=":robot:"
)

st.header("PrivateGPT")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text 

user_input = get_text()

def writeText(output):
    st.session_state.generated.append(output)

if user_input:

    with torch.inference_mode():
      
        st.session_state.past.append(user_input)
        st.session_state.generated.append(conversation_with_summary.predict(input = user_input))

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
