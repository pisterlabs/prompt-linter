
import os
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader


def run():
    st.set_page_config(
    page_title="Response to Call 21 from Gen AI Technologies", page_icon="📖")
    st.title("Gen-AI's Response to Call 21")
    st.subheader("How might we leverage Generative AI technologies (such as GPT/Large Language Models) to develop innovative solutions on OneMap, which will offer existing and new users greater convenience, as well as personalised and useful map-based information?")


    """
    The following is a chat interface that you can interact with the proposal.
    You can ask questions such as 'Who are the Problem Solvers?' or 'Is the proposal scalable?'
    or 'What is the design pattern used?', etc. Using Generative AI, the chatbot will answer questions 
    as best as it can, based on the proposal submitted. For more specific answers that is outside of the
    proposal, please contact the team directly.
    """
    
    os.environ['OPENAI_API_KEY'] = "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"
    # openai.log = "debug"
    
    # # Embeddings
    embedding_function = OpenAIEmbeddings()
    
    # Create embeddings from text
    # loader = TextLoader('./summary v4.txt')
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
    # docs = text_splitter.split_documents(documents)
    # db_connection = Chroma.from_documents(
    #     docs, embedding_function, persist_directory='./proposal_v4_db')
    # db_connection.persist()
    
    # Load document from embeddings
    db_connection = Chroma(persist_directory='./proposal_v4_db',
                           embedding_function=embedding_function)
    
    
    # Set up memory
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")
    
    # Prompt templates
    system_text = "You are a friendly chatbot that is answering questions about a proposal submission. Always add a disclaimer 'For more details and clarifications, please refer to the proposal or contact the Problem Solvers.'"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_text)
    
    human_text = "{request}\n\nYour answer should be less than 200 words and include the following context as part of your answer: {background_context}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_text)
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])
    
    
    llm = ChatOpenAI(temperature=1.5, max_tokens=1024, model="gpt-3.5-turbo-16k")
    
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
    
    # If user inputs a new prompt, generate and draw a new response
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)
        similar_docs = db_connection.similarity_search(prompt)
        background_context = []
        for similar in similar_docs:
            background_context.append(similar.page_content)
        request = chat_prompt.format_prompt(request=prompt, background_context="\n".join(
            background_context), memory=memory).to_messages()
        response = llm(request)
        st.chat_message("ai").write(response.content)
        msgs.add_ai_message(response.content)


if __name__ == "__main__":
    run()
