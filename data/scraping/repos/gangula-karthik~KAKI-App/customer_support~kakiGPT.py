from langchain import PromptTemplate, LLMChain
from firebase_admin import credentials
from firebase_admin import db
from flask_socketio import SocketIO, send
import pyrebase
import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
import logging

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]



logging.info("Starting FAQ generation...")

config = {
    "apiKey": load_dotenv("PYREBASE_API_TOKEN"),
    "authDomain": "kaki-db097.firebaseapp.com",
    "projectId": "kaki-db097",
    "databaseURL": "https://kaki-db097-default-rtdb.asia-southeast1.firebasedatabase.app/",
    "storageBucket": "kaki-db097.appspot.com",
}


firebase = pyrebase.initialize_app(config)
pyredb = firebase.database()
pyreauth = firebase.auth()
pyrestorage = firebase.storage()

logging.info("Firebase initialized.")

memory = ConversationBufferMemory(memory_key="chat_history")

# flowGPT for prompt templates

# See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
repo_id = "tiiuae/falcon-7b-instruct"
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.6, "max_new_tokens": 2000}, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

template = """
Question: {question}

Context: Take a deep breath and answer the question by thinking step by step. Provide a concise and accurate response using your extensive knowledge as a customer support representative of KAKI. If additional and relevant context is required, such as helpdesk ticket information, consider using the comments and description for a better answer.Format the answer in bullet points or paragraph and add a newline character at the end of each bullet point or paragraph.

{formatted_template}

Answer:
"""

prompt_template = PromptTemplate(template=template, input_variables=["question", "formatted_template"])


llm_chain = LLMChain(prompt=prompt_template, llm=falcon_llm)


def generate_answers(prompt):
    print("1. Started to think...")
    all_tickets_data = ""

    tickets = pyredb.child('tickets').get().val()

    for ticket in tickets:
        subject = pyredb.child(f'tickets/{ticket}/subject').get().val()
        description = pyredb.child(f'tickets/{ticket}/descriptions').get().val()
        comments = pyredb.child(f'comments/').get().val()
        comments_text = '\n'.join(comments) if comments else ''
        all_tickets_data += f"\nSubject: {subject}\nDescription: {description}\nComments: {comments_text}\n"

    
    
    formatted_prompt = {
        "question": prompt,
        "formatted_template": all_tickets_data
    }


    memory_vars = memory.load_memory_variables({})
    chat_history = memory_vars.get("chat_history", "")

    print("2. Memory has been initialized...")

    formatted_prompt = {
        "question": prompt,
        "formatted_template": all_tickets_data + "\n\nPrevious Interactions:\n" + chat_history
    }

    response = llm_chain.run(formatted_prompt)

    memory.chat_memory.add_user_message(prompt)
    if 'Answer' in response:
        memory.chat_memory.add_ai_message(response['Answer'])

    print(f"3. Questions have been answered.")

    return response




if __name__ == "__main__": 
    faqs = generate_answers("how to solve the email sync issue?")
    print(faqs)