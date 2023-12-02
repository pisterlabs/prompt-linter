import streamlit as st
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from src.html_templates import user_template, bot_template


def get_vector_embed(text_chunks, device):
    """
    This function uses Embeddings from HFH in order to convert
    the text chunks that were outputted from
    data_extraction.get_chunks() into vector representation.

    :param text_chunks: The output of get_chunks function from
    the data extraction module.
    :param device: In order to speed up processing of the module,
    either cuda (gpu) or cpu can be used
    :return: The output returns a vector embedding of the text chunks.
    Either OpenAI or HFH can be used.
    """

    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vec_store = FAISS.from_texts(texts=text_chunks,
                                 embedding=embeddings)
    return vec_store


def create_chain(vec_store):
    """
    From the vector representation, we create a conversational chain + memory
    so history can be accessed and interaction between the model and humans
    can start.

    :param vec_store: The vector embeddings yielded as output in
    get_vector_embed()
    :return: The output is the conversational chain as a json object
    after giving input.
    """

    llm_model = OpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True)

    convo_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=vec_store.as_retriever(),
        memory=memory
    )

    return convo_chain


def handle_userinput(question):
    """
    This function is the graphical part where the human asking
    will be under the human label while the bot answering will
    be under the bot label. This conversation is also persistent
    and the state is saved.

    :param question: The question asked by the human about the PDF
    :return: The output is the response given by the bot
    with respect to the PDF.
    """

    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content),
                     unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content),
                     unsafe_allow_html=True)
    st.write(response)
