from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
import gradio as gr
from llama_index import SimpleDirectoryReader
from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
import time
from llama_index.prompts  import Prompt
from llama_index import StorageContext, load_index_from_storage
from supabase import create_client, Client
import uuid
import os


def load_data():

        reader = SimpleDirectoryReader(input_dir="./documents", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

def create_custom_chatEngine(index):
   
    # list of `ChatMessage` objects
 
    template = (
    "Following Informations : \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Please answer the question always from the first person perspective and always start your answer with David: {query_str}\n"
)
    qa_template = Prompt(template)


    query_engine = index.as_query_engine(text_qa_template=qa_template)
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        verbose=False
    )
    return chat_engine

def insertTable(sessionId, text): 

    print(supabase.table('chatlog').upsert({"session_id": sessionId, "history": text}).execute())
  
# Add here your secret API-KEY from Supabase 
# For security reasons please create a enviroment variable for it 
key = os.environ['DB_KEY'] = "PASTE_YOUR_SUPABASE_DB_KEY_HERE"

# Replace with your OpenAI API key
# For security reasons please create a enviroment variable for it 

#os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_OPENAI_KEY_HERE"  

# Add here your supabase-URL
url = "PASTE_YOUR_URL_HERE"

supabase: Client = create_client(url, key)

# Create the index
index = load_data()

# Persist index
index.storage_context.persist("index_files")

#Load index from disk
storage_context = StorageContext.from_defaults(persist_dir="index_files")
index = load_index_from_storage(storage_context)

with gr.Blocks() as demo:
    chat_engine = create_custom_chatEngine(index)

    # Create new session_id
    session_id = str(uuid.uuid1())

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="⏎ for sending",
            placeholder="Ask me something",)
    clear = gr.Button("Delete")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        bot_message = chat_engine.chat(user_message)
        history[-1][1] = ""
        for character in bot_message.response:
            history[-1][1] += character
            time.sleep(0.01)
            yield history
        # insert the chat-history to our table
        insertTable(session_id,history)


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=True).then(
        bot, chatbot, chatbot
    )
    
    clear.click(lambda: None, None, chatbot, queue=True)

demo.queue(concurrency_count=1).launch(share=True)
    


