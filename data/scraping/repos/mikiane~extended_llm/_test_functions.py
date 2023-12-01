
import openai
import json
import os
from dotenv import load_dotenv
import lib__embedded_context
import os.path
from openpyxl import load_workbook
from markdownify import markdownify as md
from urllib.parse import urlparse, urljoin
from lib__path import *
from flask import Flask, request, jsonify




load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_context(text, model):
    """
    Extraire un contexte de 'text' basé sur la limite spécifiée.

    Si la longueur de 'text' est inférieure à 'limit', renvoie le texte complet.
    Sinon, renvoie une combinaison des premiers et derniers caractères de 'text'
    avec ' [...] ' inséré au milieu pour indiquer la coupure.

    :param text: La chaîne de caractères à traiter.
    :param limit: La limite de longueur pour le contexte extrait.
    :return: La chaîne de caractères traitée.
    """
    token_nb = 2000
    
    if model == "claude-2":
        token_nb = 100000 
    if model == "gpt-4":
        token_nb = 8000
    if model == "gpt-4-1106-preview":
        token_nb = 4000
    if model == "gpt-3.5-turbo-16k": 
        token_nb = 16000
    if model == "hf":
        token_nb = 2000        
    
    limit = int((int(token_nb)*3)/2)
    
    if len(text) < limit:
        return text
    else:
        half_limit_adjusted = limit // 2 - 4
        return text[:half_limit_adjusted] + ' [...] ' + text[-half_limit_adjusted:]


model = "gpt-4-1106-preview"

##################################################
def grab_content(url):    
    content = lib__embedded_context.get_text_from_url(url)
    return content

##################################################
def find_context(url, texte):
    ## formate le texte en fonction du modèle utilisé
    texte = extract_context(texte, model)
    
    # si le brain_id a la forme d'une url (http ou https), on crée un index specifique à l'url    
    index_filename= "datas/" + lib__embedded_context.build_index_url(url) + "/emb_index.csv"
    
    context = lib__embedded_context.find_context(texte, index_filename, n_results=3)
    return context
    """
    res = [{'id':1,'request':'searchcontext','answer':context}]
    response = jsonify(res)
    response.headers['Content-Type'] = 'application/json'
    return(response)
    """


##################################################
def get_news(sujet):
    ## traitement récupération des news
    content = "news about " + sujet
    return content


##################################################
def build_newsletter(contenu):
    ## traitement construction de la newsletter
    newsletter = "NL about " + contenu
    return newsletter


tools = [
    {
        "type": "function",
        "function": {
            "name": "grab_content",
            "description": "Récupérer le contenu dans la page derriere l'url transmise",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "l'url du site à parser",
                    },
                    "content": {
                        "type": "string",
                        "description": "le contenu récupéré",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_context",
            "description": "Récupérer le contexte pertinent par rapport  au texte transmis dans l'url transmise",
            "parameters": {
                "type": "object",
                "properties": {
                    "texte": {
                        "type": "string",
                        "description": "Le texte transmis qui sera utilisé pour trouver le contexte",
                    },
                    "url": {
                        "type": "string",
                        "description": "l'url du site à parser",
                    },
                    "contexte": {
                        "type": "string",
                        "description": "le contexte pertinent",
                    },
                },
                "required": ["url"],
                "texte": ["texte"],
            },
        },
    }
]






##################################################




    # Initialiser les messages
messages = []

    
# Boucle du chatbot
while True:
    # Obtenir l'entrée de l'utilisateur
    user_input = input("Vous: ")
    if user_input.lower() == 'quitter':
        print("Chatbot terminé.")
        break
    
    messages = [
    {
        "role": "user",
        "content": user_input,
    }
    ]    
    
        
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
        )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    messages.append(response_message)

    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            function_response = globals()[function_name](**function_args)

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

    for message in messages:
        if isinstance(message, dict) and message["content"] is None:
            message["content"] = ""
        if hasattr(message, "content") and message.content is None:
            message.content = ""
    
    # Ajouter l'entrée de l'utilisateur aux messages
    messages.append({"role": "user", "content": user_input})
    
    # Faire l'appel API
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0,
        stream=True
    )
    
    # Traiter la réponse
    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=True)

    
    print("\n\n\n")
##################################################











