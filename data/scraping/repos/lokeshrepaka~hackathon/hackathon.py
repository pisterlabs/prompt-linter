import streamlit as st
from streamlit_chat import message
from utils import get_initial_message, get_chatgpt_response, update_chat
import os
from dotenv import load_dotenv
import openai
from streamlit_option_menu import option_menu
import pickle
#from PyPDF2 import PdfReader
#from streamlit_extras.add_vertical_space import add_vertical_space
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback





load_dotenv()



with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)




with st.sidebar:
    
    selected = option_menu('GEN LEARN',
                          
                          ['GEN TUTOR'],
                    
                          icons=['person','palette'],
                          default_index=0)
    
if (selected == 'GEN TUTOR'):

    openai.api_key = os.getenv('OPENAI_API_KEY')

    st.title("Gen Learn : Chatbot")
    st.subheader("Use our GEN AI Tutor:")

    model = st.selectbox(
        "Select a model",
        ("gpt-3.5-turbo",)
    )

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    query = st.text_input("Query: ", key="input")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = get_initial_message()
    
    if query:
        with st.spinner("generating..."):
            messages = st.session_state['messages']
            messages = update_chat(messages, "user", query)
            # st.write("Before  making the API call")
            # st.write(messages)
            response = get_chatgpt_response(messages,model)
            messages = update_chat(messages, "assistant", response)
            st.session_state.past.append(query)
            st.session_state.generated.append(response)
            
    if st.session_state['generated']:

        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))



# if (selected == 'GEN SCOLOR'):
#     st.header("GEN SCOLAR")
 
 
#     # upload a PDF file
#     pdf = st.file_uploader("Upload your PDF", type='pdf')
 
#     # st.write(pdf)
#     if pdf is not None:
#         pdf_reader = PdfReader(pdf)
        
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
 
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#             )
#         chunks = text_splitter.split_text(text=text)
 
#         # # embeddings
#         store_name = pdf.name[:-4]
#         st.write(f'{store_name}')
#         # st.write(chunks)
 
#         if os.path.exists(f"{store_name}.pkl"):
#             with open(f"{store_name}.pkl", "rb") as f:
#                 VectorStore = pickle.load(f)
#             # st.write('Embeddings Loaded from the Disk')s
#         else:
#             embeddings = OpenAIEmbeddings()
#             VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
#             with open(f"{store_name}.pkl", "wb") as f:
#                 pickle.dump(VectorStore, f)
 
#         # embeddings = OpenAIEmbeddings()
#         # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
#         # Accept user questions/query
#         query = st.text_input("Ask questions about your PDF file:")
#         # st.write(query)
 
#         if query:
#             docs = VectorStore.similarity_search(query=query, k=3)
 
#             llm = OpenAI()
#             chain = load_qa_chain(llm=llm, chain_type="stuff")
#             with get_openai_callback() as cb:
#                 response = chain.run(input_documents=docs, question=query)
#                 print(cb)
#             st.write(response)



