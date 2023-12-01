import streamlit as st 
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
#from langchain.chat_models import ChatOpenAI
#from langchain.llms import OpenAI
import tempfile
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.document_loaders.csv_loader import CSVLoader

def main():
    load_dotenv()
    st.set_page_config(page_title="ask your CSV questions")
    st.header("Ask your CSV")


    user_csv=st.file_uploader("Upload your csv file",type="csv")
    if user_csv :
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(user_csv.getvalue())
            tmp_file_path = tmp_file.name

        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        data = loader.load()

        embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectors = FAISS.from_documents(data, embeddings)

        user_question=st.text_input("Ask a question about your CSV:")
        
        #llm=OpenAI(temperature=0.8)

        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":3000})
        
        conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectors.as_retriever())

        
        #agent=create_csv_agent(llm,user_csv,verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

        def conversational_chat(query):
        
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
        
            return result["answer"]
    
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]
            
        #container for the chat history
        response_container = st.container()
        #container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                    



if __name__=="__main__":
    main()
