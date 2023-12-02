from langchain import PromptTemplate, LLMChain
import asyncio
import firebase_admin
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
import logging

try:
    load_dotenv(find_dotenv())
    HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
except KeyError:
    print("Error: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
except Exception as e:
    print(f"An unknown error occurred ({e})...")

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


# See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
repo_id = "tiiuae/falcon-7b-instruct"
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.6, "max_new_tokens": 2000}, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

template = """Question: {question}

Using this information about helpdesk tickets: {formatted_template}

Give a concise answer to the question around 3 to 4 lines. DO NOT mention the ticket ids at all. Give the most important tickets or answers that apply to the most important ticket in your opinion. Reply as a customer support agent who is providing high quality support to the customers. Your main objective is to help users solve their problems.

Answer: Let's think step by step."""
prompt_template = PromptTemplate(template=template, input_variables=["question", "formatted_template"])



llm_chain = LLMChain(prompt=prompt_template, llm=falcon_llm)

def generate_faqs():
    faqs = []
    logging.info("Generating FAQs...")
    all_tickets_data = ""

    tickets = pyredb.child('tickets').get().val()

    if not tickets:
        return faqs

    for ticket in tickets:
        subject = pyredb.child(f'tickets/{ticket}/subject').get().val()
        description = pyredb.child(f'tickets/{ticket}/descriptions').get().val()
        comments = pyredb.child(f'comments/').get().val()
        comments_text = '\n'.join(comments) if comments else ''
        all_tickets_data += f"\nTicket ID: {ticket}\nSubject: {subject}\nDescription: {description}\nComments: {comments_text}\n"

    prompts = [
        "What are the most common issues reported?",
        "Which tickets are still open?",
        "What is the best solution for any issue?",
        "What are the next steps for resolving open tickets?"
    ]
    logging.info("Prompts have been loaded...")

    i = 1
    for prompt in prompts:
        formatted_prompt = {
            "question": prompt,
            "formatted_template": all_tickets_data
        }
        response = llm_chain.run(formatted_prompt)
        faqs.append({
            "question": prompt,
            "answer": response
        })
        logging.info(f"FAQ {i} has been generated.")
        i += 1


    logging.info(f"FAQs have been generated.")
    return faqs




if __name__ == "__main__": 
    faqs = generate_faqs()
    print(faqs)