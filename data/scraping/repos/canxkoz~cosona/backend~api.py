from flask import Flask, request
import cohere
from pymongo import MongoClient
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import pprint
import os
from constants import ALLOWED_EXTENSIONS, PROMPTS, NEW_PROMPTS
from db import store_pdf
from langchain.chat_models import ChatCohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from flask_cors import CORS
from langchain.chat_models import ChatCohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatCohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import json

load_dotenv()
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://cosona.vercel.app"])
dbClient = MongoClient(os.getenv("ATLAS_URI"))
cohere_db = dbClient[os.getenv("DB_NAME")]
printer = pprint.PrettyPrinter(indent=4)
co = cohere.Client(os.getenv("COHERE_API_KEY"))
chat_model = ChatCohere(
    cohere_api_key=os.getenv("COHERE_API_KEY"), model="command", temperature="0.5"
)


@app.route("/")
def hello_world():
    return "Hello, World!"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route("/api/upload", methods=["GET", "POST"])
# def upload_script():
#     if request.method == "POST":
#         if "file" not in request.files:
#             printer.pprint("No file part")
#         file = request.files["file"]
#         # If the user does not select a file, the browser submits an
#         # empty file without a filename.
#         if file.filename == "":
#             printer.pprint("No selected file")
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             store_pdf(cohere_db, file, filename)
#             printer.pprint(f"File saved {filename}")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    chat_history = data["text"]
    character = data["character"]
    if character == "":
        return {"error": "Please select a character"}
    response = get_response(chat_history, character)

    return response
    
def get_character_list(text):
    user_input2 = """List the main cast characters from the movie script. Answer only this this FORMAT:
                The list of characters are:
                    -Character1 Full name
                    -Character2 Full Name
                    -Character3 Full Name
                    -Character4 Full Name
                    -Character5 Full Name
                    Who would you like to talk to? """
    # Define the character's name and description
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n"],chunk_size=2500, chunk_overlap=300)
    texts = text_splitter.split_text(text)
    # Initialize the chat model for the character
    chat_model = ChatCohere(cohere_api_key=cohere_api_key, model='command-light',temperature=0.0)
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
    # Create a vectorstore from documents
    vectorstore = Chroma.from_texts(texts, embeddings)
    # Create retriever interface
    retriever=vectorstore.as_retriever()
    # Create the memory object
    memory=ConversationBufferMemory(
      memory_key='chat_history', return_messages=True)
    #prompt
    custom_prompt_template = """
                    {context}
                    You are a character extracter bot. You're only job is to extract the main cast character names from a movie script.
                    Main cast characters will have more than 15 lines of dialogue. Do not include character with less than 10 dialogues in the character list
                    Extract the list of 5-6 main characters only from the movie script in the format
                    FORMAT:
                    The list of characters are:
                        -Character1 Full name
                        -Character2 Full Name
                        -Character3 Full Name
                        -Character4 Full Name
                        -Character5 Full Name
                        Who would you like to talk to? 
    
                    Always end the chat with "Who would you like to talk to?"
    
                    
              
            """
    
    prompt = PromptTemplate(template=custom_prompt_template,
                    input_variables=['context', 'question'])
    #list retrieval chain
    chain = RetrievalQA.from_chain_type(llm=chat_model,
                                    chain_type='stuff',
                                    retriever=retriever,
                                    return_source_documents=False,
                                    chain_type_kwargs={'prompt': prompt}
                                    )
    response=chain({'query': user_input2})
    
    list = response["result"]
    return list

def get_response(messages, character):
    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(PROMPTS[character]),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),  # last message
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation = LLMChain(
        llm=chat_model, prompt=prompt_template, verbose=True, memory=memory
    )

    response = conversation.invoke({"input": PROMPTS[character] + messages[-1]})

    # Return the response
    res = {
        "message": response["text"],
    }

    return res


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
