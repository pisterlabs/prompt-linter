import streamlit as st
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
# from pydub import AudioSegment
import speech_recognition as sr

def get_audio_text(audio_file):
    text = 'This text is generated from a audio file. '
    # # convert mp3 file to wav                                                       
    # sound = AudioSegment.from_mp3("transcript.mp3")
    # sound.export("transcript.wav", format="wav")

    # Initialize recognizer class (for recognizing the speech)
    r = sr.Recognizer()
    
    # Reading Audio file as source
    # listening the audio file and store in audio_text variable

    with sr.AudioFile(audio_file) as source:
    
        audio_text = r.record(source)
    
        # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
        try:
            # using google speech recognition
            text += r.recognize_google(audio_text)
            # print(text)
        except sr.RequestError as e:
            st.write(e)
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with Audio", page_icon=":microphone:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header("Chat with Audio :microphone:")
    user_question = st.text_input("Ask a question about your Audio:")
    if user_question:
        handle_userinput(user_question)
        
    with st.sidebar:
            
        st.subheader("Your Audio")
        audio_file = st.file_uploader("Upload your Audio here and click on 'Process'", accept_multiple_files=False, type=['wav'])
        
        if st.button("Process"):
            if audio_file:
                with st.spinner("Processing"):
                    try:
                        # get pdf text
                        raw_text = get_audio_text(audio_file)

                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # Update the processing status in the sidebar
                        st.sidebar.info("Processing completed.")

                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore) 
                    except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.error("Please upload an audio file.")

if __name__ == '__main__':
    main()