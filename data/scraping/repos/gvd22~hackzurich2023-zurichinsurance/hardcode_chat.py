import inspect

import openai
import json
import os
from datetime import datetime

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
from dotenv import load_dotenv
from langchain import embeddings
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client.grpc import PointStruct

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")
openai_api_key = os.getenv("OPENAI_KEY")
user_chat_history = []

context = ""

from qdrant_client import QdrantClient

# qdrant_client = QdrantClient(
#     url="hackzurich23-vectordb-emcg5a6iia-oa.a.run.app",
#     api_key="<your-api-key>",
# )

#qdrant_client = QdrantClient('https://hackzurich23-vectordb-emcg5a6iia-oa.a.run.app')

location = "https://hackzurich23-vectordb-emcg5a6iia-oa.a.run.app"
port = 443

client = QdrantClient(location=location, port=port)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


def aktuelles_datum_formatiert():
    jetzt = datetime.now()
    return jetzt.strftime('%A, %B %d, %Y %H:%M')

# Das aktuelle Datum in einer Variable speichern
datum_und_uhrzeit = aktuelles_datum_formatiert()


def get_current_status_of_claim(information):
    status_info = {
        "status": "accepted",
    }
    return json.dumps(status_info)

def check_args(function, args):
    sig = inspect.signature(function)
    params = sig.parameters

    # Check if there are extra arguments
    for name in args:
        if name not in params:
            return False
    # Check if the required arguments are provided
    for name, param in params.items():
        if param.default is param.empty and name not in args:
            return False

    return True



def qdrant_search_api(seacrchquestion):
    embedded_query = embeddings.embed_query(seacrchquestion)
    hits = client.search(
        collection_name="zurizap",
        query_vector=embedded_query,
        limit=10  # Return 5 closest points
    )
    prompt = ""
    for result in hits:
        prompt += result.payload['text']

    concatenated_answer = " ".join([prompt])

    return  concatenated_answer


functions = [
    {
        "name": "get_current_status_of_claim",
        "description": "Get the current Status for the claim from the user",
        "parameters": {
            "type": "object",
            "properties": {
                "information": {
                    "type": "string",
                },
            },
            "required": ["information"],
        },
    },
    {
        "name": "get_data_from_qdrant",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                },
            },
            "required": ["question"],
        },
    }
]

available_functions = {
            "get_current_status_of_claim": get_current_status_of_claim,
        }

messages = []

def run_conversation(user_input,functions,available_functions):
    context = qdrant_search_api(user_input)

    messages.append({"role": "system",
         "content":
             "You are a Customer Supporter for the Zurich Insurance and your Name ist ZüriZap."
             "Your job is to answer questions about your client's insurance policies."
             "You the AI were trained on data until 2021 and has no knowledge of events that have taken place since then. "
             "You also have no way of accessing data on the internet, so you should not claim you can or say you will look. "
             "Try to formulate your answers concisely, although this is not necessary. "
             f"Closing date: Saturday, January 1, 2022 / Current date: {datum_und_uhrzeit}"
             "ZüriZap is the digital assistant of the Zurich Insurance "
             "ZüriZap is a chatbot that answers questions Insurance policies"
             "All questions on this topic ZüriZap must always ground on the basis of information from the Knowledgebase."
             "Answers should always be answered on the basis of the information provided to you. "
             "It is important that the answer be short and precise."
             "YOU SHOULD ALWAYS ANSWER BASED ON THIS INFORMATION IF YOU DONT HAVE ENOUGH INFORMATION THEN ASKE A QUESTION TO THE QDRANT DATABASE:"
             f"{context}"
         })


    # Step 1: send the conversation and available functions to GPT
    messages.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
          # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]

        if function_name not in available_functions:
            return "Function " + function_name + " does not exist"
        function_to_call = available_functions[function_name]

        # verify function has correct number of arguments
        function_args = json.loads(response_message["function_call"]["arguments"])
        if check_args(function_to_call, function_args) is False:
            return "Invalid number of arguments for function: " + function_name
        function_response = function_to_call(**function_args)

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response

        response = second_response['choices'][0]['message']['content']
        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )
        return response
    else:
        chat_message = response['choices'][0]['message']['content']
        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )
        return chat_message

#print(run_conversation(functions,available_functions))

def chatbot():

    print("Bot: Hey I'm ZüriZap how can I help you?")
    user_input = input("User: ")
    user_chat_history.append({"role": "assistant", "content": "Hey I'm ZüriZap how can I help you?" })
    user_chat_history.append({"role": "user", "content": user_input})

    while True:
        print("Bot: ", end="")
        print(run_conversation(user_input,functions,available_functions))

        user_input = input("User:")


chatbot()