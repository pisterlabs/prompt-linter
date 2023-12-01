# main.py
"""
Interactive chat with memory for earlier chats.
"""
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin  # Setting cors for http
from index_handler import load_index,store_index
from llama_index.memory import ChatMemoryBuffer
import logging
import asyncio
import threading
import re
from utils import check_nature_of_my_response,handle_user_prompt
from llama_index.llms import Anthropic
from llama_index import ServiceContext
from llama_index.query_engine import Re
# Create a Flask app
chat_app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for the app
cors = CORS(chat_app)  # Setting cors for http
chat_app.config["CORS_HEADERS"] = "Content-Type"
logging.basicConfig(level=logging.DEBUG)
text_completion=""
# Define a function to run in a thread
import re
import json

def handle_chat_thread(chat_engine, input_text):
    global text_completion
    text_completion = chat_engine.chat("Only answer from the context," + input_text)
    
    if hasattr(text_completion, 'source_nodes'):
        document_info = str(text_completion.source_nodes)
        
        # Extract URL, department, section, and title from the JSON data
        urls = re.findall(r'"url": "([^"]*)"', document_info)
        departments = re.findall(r'"department": "([^"]*)"', document_info)
        sections = re.findall(r'"section": "([^"]*)"', document_info)
        titles = re.findall(r'"title": "([^"]*)"', document_info)
        
        print("*" * 100, "\n", document_info)
        
        if urls:
            # Create a list of dictionaries containing all four attributes
            info_list = [
                {
                    "url": url,
                    "department": department,
                    "section": section,
                    "title": title,
                }
                for url, department, section, title in zip(urls, departments, sections, titles)
            ]
            
            print('\n' + '=' * 60 + '\n')
            print('Context Information:')
            for info in info_list:
                print(info)
            print('\n' + '=' * 60 + '\n')
        
        
    else:
        print("no such attribute^^^^^")

"""
Method for handling user chat api calls # Define an API route for handling user text input
"""

@chat_app.route("/api/send_text", methods=["POST"])
@cross_origin()  # Setting cors for http
def send_text():
    try:
        message_response = "Server Error or Usage Limit"  # To avoid HTTPSConnectionPool(host='api.openai.com', port=443): Read timed out
        data = request.get_json()
        input_text = (
            data.get("text", "hi") + "?"
        )  # if no text in json,add 'hi' and add '?' to make proper conversation answer.
        flag = False
        message_response,flag=handle_user_prompt(input_text)
        if not flag:
            # Load the chat engine
            if not hasattr(chat_app, "chat_engine") or chat_app.chat_engine is None:
                asyncio.run(get_chat_engine())
                logging.info(
                    "Chat engine was missing - Recreated."
                )
            #making a thread for handling openai chat calls
            chat_thread = threading.Thread(target=handle_chat_thread,kwargs={"chat_engine":chat_app.chat_engine,"input_text":input_text})
            chat_thread.start()
            #main thread waiting for chat_thread completion
            chat_thread.join(30)
            #assigning the value created by chat_thread
            if text_completion is not None:
                message_response = check_nature_of_my_response(f"{text_completion}")
                if len(message_response) == 0:
                    message_response = "Inconvenience Regretted!\nNo response could be generated..!"
    except Exception as e:
        logging.error("Exception occurred while generating the response - " + str(e))
        message_response = "Inconvenience Regretted!\nAn Exception occurred while generating the response."
    finally:
        return jsonify({"response": message_response})

"""
api call for resetting the chat # Define an API route for resetting the chat (if needed)
"""

@chat_app.route("/api/chat_reset", methods=["POST"])
@cross_origin()  # Setting cors for http
def chat_reset():
    if hasattr(chat_app, "chat_engine") and chat_app.chat_engine is not None:
        chat_app.chat_engine.reset()
        response = "Chat Resetted"
    else:#when chat_enine not exist or None
        #creating a new chat_engine
        asyncio.run(get_chat_engine())
        response = (
             "Chatbot is Ready..!"
        )
        logging.error( "Chatbot is Ready..!")

    return make_response(response, 200)


"""
Method for loading the index file and initializing the chat_engine with memory
"""
async def get_chat_engine():
    logging.info(" Inside method get_chat_engine ")
    if not hasattr(chat_app, "chat_engine"):
        index = load_index()  # Load your index using your custom function
        service_context = ServiceContext.from_defaults()
        memory = ChatMemoryBuffer.from_defaults()
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            service_context=service_context,
            similarity_top_k=2,
            memory=memory,
            system_prompt="You are a virtual office assistant of Botswana Government with a limited knowledge to "
                      "given context, you should respond to user queries with a friendly language, "
                      "you should engage in small talk, you should be empathetic, who can provide clear "
                      "and helpful responses."
                      "\nAsk user to clarify the question if you are confused or have multiple answers."
                      "\nYou can only provide answers and information based on the PROVIDED CONTEXT. "
                      "\nIf you did not find the relevant information in the context, please respond to "
                      "the user that: 'I'm afraid that what you asked is something that I can't answer "
                      "per my knowledge. Sorry for the inconvenience caused. Please let me know if "
                      "there is something else that I can help you with.'"
                      "\nWhen you respond to a user, never mention that you are answering from 'provided context'."
        )
        chat_app.chat_engine = chat_engine  # Store chat_engine in the app context


# Run the Flask app if this script is the main entry point
if __name__ == "__main__":
    # asyncio.run(store_index()) # Called only once to create the index files
    asyncio.run(get_chat_engine())
    chat_app.run(host="0.0.0.0", port=5000,debug=True)
