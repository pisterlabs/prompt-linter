import streamlit as st
from langchain.document_loaders import DirectoryLoader, UnstructuredEmailLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import sys
sys.path.append("../")
from htmlTemplates import css, bot_template, user_template

# loader = UnstructuredEmailLoader(
#     "example_data/fake-email.eml",
#     mode="elements",
#     process_attachments=True,
# )

def get_emails():
    emails = []
    kwargs = {"mode": "elements"}
    # loader = DirectoryLoader(
    #     'emails/', 
    #     glob='**/*.eml', 
    #     show_progress=True, 
    #     use_multithreading=True, 
    #     loader_cls=UnstructuredEmailLoader,
    #     loader_kwargs=kwargs
    # )
    loader = UnstructuredEmailLoader(
        "emails/File.eml",
        mode="elements", 
        process_attachments=True
    )    
    print(loader.load_and_split())
    loader = UnstructuredEmailLoader(
        "emails/File.eml"
    )
    print(loader.load())
    # emails.extend(loader.load())  
    # loader = UnstructuredEmailLoader(
    #     "emails/Doc384714_78718.eml",
    #     mode="elements"
    # )
    emails.extend(loader.load())   
    return emails 


def get_email_chunks(email):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    # text_splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len
    # )
    chunks = text_splitter.split_documents(email)
    chunks = filter_complex_metadata(chunks)
    # st.write(chunks)
    return chunks

def get_chroma_vectorstor(email_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(email_chunks, embeddings, collection_name='multipleemails')
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=1, request_timeout=60, verbose=True)

    template = """Given the following conversation respond to the best of your ability in a 
    professional voice and act as an insurance expert explaining the answer to a novice
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    template = """Given the following conversation respond as an insurance expert, rephrase 
    the follow up question to be a standalone question and explain 
    clearly the answer to a novice insurance employee and respond in french.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""


    QA_PROMPT_DOCUMENT_CHAT = """You are a helpful and courteous support representative working for an insurance company. 
    Use the following pieces of context to answer the question at the end.
    If the question is not related to the context, politely respond that you are tought to only answer questions that are related to the context.
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer. 
    Try to make the title for every answer if it is possible. Answer in markdown.
    Make sure that your answer is always in Markdown.
    {context}
    Question: {question}
    Answer in HTML format:"""

    CONDENSE_PROMPT = """Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question and respond in english.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    QA_PROMPT = PromptTemplate(
        input_variables=['context', 'question'], 
        template=QA_PROMPT_DOCUMENT_CHAT
    )
    CONDENSED_PROMPT = PromptTemplate(
        input_variables=['chat_history','question'],
        template=CONDENSE_PROMPT
    )

    PROMPT = PromptTemplate(
        input_variables=["chat_history", "question"], 
        template=template
    )

    # memory = ConversationBufferMemory(
    #     memory_key='chat_history', return_messages=True, output_key='answer')
    memory = ConversationBufferWindowMemory(
        k=1, 
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer'
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        # condense_question_prompt=PROMPT,
        return_source_documents=True,
        verbose=True,
        condense_question_prompt=CONDENSED_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        verbose=True
    )
    # st.write(conversation_chain)
    return conversation_chain

def handle_userinput(user_question):
    chat_history = [] # st.session_state.chat_history
    # print(chat_history)
    bot_message = ''
    references = ''
    response = st.session_state.conversation({'question': user_question, "chat_history": chat_history})
    st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            bot_message = bot_template.replace("{{MSG}}", message.content)
            bot_message = bot_message.replace("{{REFERENCES}}", references)
            st.write(bot_message, unsafe_allow_html=True)
    if 'question' in response:
        st.write(user_template.replace(
            "{{MSG}}", response['question']), unsafe_allow_html=True)
    if 'answer' in response:
        bot_message = bot_template.replace("{{MSG}}", response['answer'])
        references += '<br><h5>References</h5>'
        references += '<ol>'
        for source_document in response['source_documents']:
            references += '<li>' + source_document.metadata['source'].replace("docs\\", "")
            if 'page' in source_document.metadata:
                references += ' (' + str(source_document.metadata['page'] + 1) + ')'
            references += '</li>'
        references += '</ol>'
        bot_message = bot_message.replace("{{REFERENCES}}", references)
        st.write(bot_message, unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple emails",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple emails :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        emails = st.file_uploader(
            "Upload your emails here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get email text
                raw_email = get_emails()

                # get the text chunks
                #text_chunks = get_text_chunks(raw_text)
                email_chunks = get_email_chunks(raw_email)

                # create vector store
                # vectorstore = get_vectorstore(text_chunks)
                vectorstore = get_chroma_vectorstor(email_chunks)
                
                # create conversation chain
                result = get_conversation_chain(vectorstore)
                print(result({'question': "What was the content of the mail?", "chat_history": []}))
                # st.session_state.conversation = get_conversation_chain(vectorstore)
                # print('1')
                # print(st.session_state.conversation)
                # print('2')
                
if __name__ == '__main__':
    main()