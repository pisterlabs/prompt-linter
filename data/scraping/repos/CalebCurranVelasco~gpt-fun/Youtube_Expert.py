from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import textwrap
import streamlit as st


load_dotenv(find_dotenv())
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

#creates the converstaion chain with langchain
def get_conversation_chain(db):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory
    )
    return conversation_chain


# def get_response_from_query(db, query, k=4):
#     """
#     gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
#     the number of tokens to analyze.
#     """

#     docs = db.similarity_search(query, k=k)
#     docs_page_content = " ".join([d.page_content for d in docs])

#     chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

#     # Template to use for the system message prompt
#     template = """
#         You are a helpful assistant that that can answer questions about youtube videos 
#         based on the video's transcript: {docs}
        
#         Only use the factual information from the transcript to answer the question.
        
#         If you feel like you don't have enough information to answer the question, say "I don't know".
        
#         Your answers should be verbose and detailed.
#         """

#     system_message_prompt = SystemMessagePromptTemplate.from_template(template)

#     # Human question prompt
#     human_template = "Answer the following question: {question}"
#     human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [system_message_prompt, human_message_prompt]
#     )

#     chain = LLMChain(llm=chat, prompt=chat_prompt)
#     # st.write(response)
   

#     response = chain.run(question=query, docs=docs_page_content)
#     response = response.replace("\n", "")
    
#     return response


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question}) # note conversation chain already has all the configuration with the vector store and memory
    # st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # this is for the picture and text boxes
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



# Example usage:
# video_url = "https://www.youtube.com/watch?v=NYSWn1ipbgg&list=PL-Y17yukoyy3zzoMJNkWQuogKbWGyBL-d"
# video_url = input("Please enter the youtube url: ")
# db = create_db_from_youtube_video_url(video_url)

# query = "Who is the person talking in the video?"
# query = input("Question for the youtube video: ")
# response, docs = get_response_from_query(db, query)
# print(textwrap.fill(response, width=50))


def main():
    
    st.set_page_config(page_title="Chat with a Youtube Video", page_icon=":movie_camera:")
    st.write(css, unsafe_allow_html=True)


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "db" not in st.session_state:
        st.session_state.db = None
    if "conversation" not in st.session_state: # initialize it here when using session state, can use this variable globally now and does not ever reset
        st.session_state.conversation = None

    st.header("	:microphone: Chat With Youtube Videos :movie_camera:")
    user_question = st.text_input("Ask a question to your youtube video:")

    if user_question: # if the user inputs a question
        handle_userinput(user_question)


        # query = user_question

        # if st.session_state.db is None:  # Check if db is not initialized
        #     video_url = st.session_state.get("URL", "")
        #     if video_url:
        #         with st.spinner("Processing"):
        #             st.session_state.db = create_db_from_youtube_video_url(video_url)
        #     else:
        #         st.error("Please enter a valid Youtube URL and click 'Process' before asking a question.")
        #         return

        # response = handle_userinput(st.session_state.db, query)
        # st.session_state.chat_history = response['chat_history']

    
    
    # this creates the sidebar to upload the pdf docs to
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Upload a youtube url\n"  
            "2. Process the video\n"
            "3. Start talking to the youtube video!"
        )

        st.markdown("---")

        st.subheader("Your video")
        video_url = st.text_input(
            "Youtube Video URL",
            placeholder="Paste your Youtube URL here",
            value=st.session_state.get("URL", ""),
        )
        st.session_state["URL"] = video_url
        if st.button("Process"):
            with st.spinner("Processing"):
                st.session_state.db = create_db_from_youtube_video_url(video_url)
                st.session_state.conversation = get_conversation_chain(st.session_state.db)
                


if __name__=='__main__':
    main()
