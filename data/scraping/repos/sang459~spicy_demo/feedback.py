import streamlit as st
import openai
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
from llama_index import load_index_from_storage, StorageContext
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.schema import Node, NodeWithScore
from llama_index.response_synthesizers import ResponseMode, get_response_synthesizer

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
storage_context = StorageContext.from_defaults(persist_dir="./data/")

index = load_index_from_storage(storage_context) 
retriever = index.as_retriever()

def generate_retriever_query(chat_history_for_display):
    """
    chat_history_for_display를 받아서, retriever query를 만들어주는 함수

    반환값

    query: str
    """
    
    chat_history_for_retrieval_query = chat_history_for_display.copy()
    chat_history_for_retrieval_query.append({"role": "user", "content": "[INSTRUCTION] To address the user's concerns based on our chat history, you will refer to the CBT guideline book for adult ADHD. Provide relevant keywords or topics of interest so you can extract the most pertinent information. ONLY KEYWORDS, in ENGLISH."})

    response = openai.ChatCompletion.create(
                    model= "gpt-4",
                    messages=chat_history_for_retrieval_query,
                    stream=False,
                    temperature=0.5,
                    top_p = 0.93
                    )

    query = response['choices'][0]['message']['content']

    return query

def generate_relevant_info(chat_history_for_display, user_input):
    print('GENERATING QUERY...')
    query = generate_retriever_query(chat_history_for_display)
    print(chat_history_for_display)
    print(query)
    nodes = retriever.retrieve(query)
    print(nodes)
    processor = SimilarityPostprocessor(similarity_cutoff=0.85)
    filtered_nodes = processor.postprocess_nodes(nodes)
    print('SYNTHESIZING MEMO...')

    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)

    i = 0
    while i < 3:
        response = response_synthesizer.synthesize(f"Provide a short, precise, straight-to-the-point and informative memo for a CBT counselor dealing with an adult ADHD patient. The patient's message was: {user_input}", nodes=filtered_nodes)
        response = response.response
        i += 1
        print(i)
        if type(response) == str:
            break
    
    if type(response) != str:
        response = ""
    
    print("GENERATED MEMO: \n"+response)

    return response


st.markdown("""
            <style>
            [data-testid="stSidebarNav"] {
                display: none
            }

            [data-testid="stSidebar"] {
                display: none
            }
            </style>
            """, unsafe_allow_html=True)

def get_response(chat_history_for_model_day2):
    print("##############GETTING RESPONSE##############")
    print(chat_history_for_model_day2)
    response = openai.ChatCompletion.create(
                model= "gpt-4",
                messages=chat_history_for_model_day2,
                stream=False,
                temperature=0.5,
                top_p = 0.93
                )
    return response['choices'][0]['message']['content']
    # 대화 시작

for message in st.session_state.chat_history_for_display_day2:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")
if user_input:
    st.session_state.chat_history_for_display_day2.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    memo = "[GUIDANCE MEMO]\n"
    memo = memo + generate_relevant_info(st.session_state.chat_history_for_display_day2, user_input)
    # Append the memo to chat_history_for_model
    st.session_state.chat_history_for_model_day2.append({"role": "user", "content": user_input + "\n" + memo})

    response = get_response(st.session_state.chat_history_for_model_day2)
    st.session_state.chat_history_for_model_day2.append({"role": "assistant", "content": response})
    st.session_state.chat_history_for_display_day2.append({"role": "assistant", "content": response})

    st.experimental_rerun()



