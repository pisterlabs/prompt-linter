# This section imports all the necessary libraries and modules. It includes tools for handling the chat interface (Gradio), embeddings, memory, chains, and loading secrets for secure communication

import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from secret_config import LoadSecrets


# Here, we instantiate and call a method to load secrets, such as API keys or other sensitive information, necessary for connecting to external services
s_in = LoadSecrets()
s_in.load_secret()

# Setting Up Embeddings and Database Configuration. This segment sets up the persistent directory and initializes embeddings using OpenAI's model. The Chroma database is configured to work with these embeddings.
persist_directory = "transcript_db"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=1)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)

# Conversation Memory Buffer to store the chat history, allowing the bot to have context-aware conversations:
memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=False,
                                  )



# Language Model Initialization configured with specific parameters, like 'temperature', to control its behavior.
llm = OpenAI(model_name="text-davinci-003", temperature=0.2, max_retries=1, max_tokens=1000)

# Building the Conversational Chain to retrieve chain that combines the language model, the retriever, and memory to form a complete conversational system

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(),
    memory=memory,
    get_chat_history=lambda h: h,
    verbose=True
)

# User Interface with Gradio:

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)
    msg = gr.Textbox()
    clear = gr.Button('Clear')

    def user(user_message, history):
        print(user_message)
        return "", history + [[user_message, None]]

    def bot(history):
        print(history)
        bot_message = qa.run({"question": history[-1][0], "chat_history": history[:-1]})
        if bot_message is None:
            bot_message = "I don't know"
        history[-1][-1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=True)


#This script sets up a conversational bot that utilizes OpenAI for natural language processing and Gradio for the user interface