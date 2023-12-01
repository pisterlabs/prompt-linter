from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models.openai import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
import os
import re
import openai
import requests
from datetime import date
from babel.dates import format_date

today = date.today()
DATE = format_date(today, format="long", locale="fr_FR")

# Charge les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupère les variables
FILE_EMBEDING = os.getenv("FILE_EMBEDING")
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("model")
VERBOSE=True
NUM_LINK = 3


def printdebug(text):
  if VERBOSE:
    print(text)

def get_webpage_text(url):
  text_content = ""
  try :
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'a', 'span']):
      text_content += tag.get_text() + "\n"
    printdebug(url+" est reussie")
  except :
    text_content = ""
    printdebug(url+" est un echec")

  print(text_content)
  printdebug("end embeding")
  return text_content

def analyze_text(prompt):
    prompt = """
    La date actuel est : """+DATE+""".Repond en français , essaie de faire une reponse complete en te basant sur tes ressources:
    """+prompt
    raw_documents = TextLoader(FILE_EMBEDING).load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    pages = text_splitter.split_documents(raw_documents)
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)
    index = VectorstoreIndexCreator().from_documents(texts)
    return index.query(prompt, llm=ChatOpenAI(model=MODEL,temperature=0))

def multi_traitement_ressource(prompt,data):
    printdebug("start embeding")
    with open(FILE_EMBEDING, 'w', encoding='utf-8') as file:
        file.write(str(data))
    result = analyze_text(prompt)
    return result

def clean_url(url):
    if url.endswith('%5D'):
        return url[:-3]
    if url.endswith('],'):
        return url[:-2]
    return url

def web_qa(url_list, query):
  content_ressource = ""
  for i in url_list:
    url = clean_url(i)
    printdebug('loading url : '+str(url))
    content_ressource += get_webpage_text(str(url))
  try :
    reponse = multi_traitement_ressource(query,content_ressource)
  except :
    reponse = "Je suis desolé mais la page web n'est pas valide"
  return "Reponse : \n"+reponse

def trouver_urls(texte):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(pattern, texte)[:NUM_LINK]

def get_prompt_template_function(demande,memoire=""):
    promptprincipale = "Reponds au besoin suivant : \n"+str(demande)
    system = "Vous êtes un assistant chargé de répondre aux questions relatives aux recherches internet.\n\nVoici le fil de la conversation :\n"+memoire
    return [{"role": "system", "content": system},{"role": "user", "content": promptprincipale}]

def get_function_openai():
    Fonction = [
        {
            "name": "search",
            "description": "useful for when you need to search information with internet",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "the title of the search",
                    }
            }
        }
        }
    ]
    return Fonction

def ChatCompletion_openai_with_function(demande,memoire):
    completion = openai.ChatCompletion.create(
        model=MODEL,
        messages=get_prompt_template_function(demande,memoire),
        functions=get_function_openai(),
    )
    return completion

def create_url(prompt):
    prompt="Tu doit transformer la demande en recherche google de mots cles : '"+prompt+"'"
    system = "Tu doit transformer la demande en recherche google de mots cles. exemple (question : 'qui as gagné le gp explorer de squeezie ?'\nReponse:https://www.google.com/search?q=winner+of+GP+Explorer+by+Squeezie). Si la question porte sur un sujet avec un repere temporel , tu peux t'aider de la date d'aujourd'hui :"+DATE+".s"
    while True:
        completion = openai.ChatCompletion.create(
            model=MODEL,
            temperature=0.3,
            messages=[{"role": "system", "content": system},{"role": "user", "content": prompt}],)
        url = trouver_urls(completion.choices[0].message.content)
        if str(url) == "[]":
            print(url)
            continue
        else:
            print(url)
            return url[0]
            break


def search_on_google(query,url):

  HEADERS = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
  }

  url = clean_url(url)
  response = requests.get(url, headers=HEADERS)
  soup = BeautifulSoup(response.content, "html.parser")
  # Récupérer tous les textes des résultats de recherche
  texts = [result.text for result in soup.find_all("div", class_="tF2Cxc")]

  reponse = multi_traitement_ressource(query,texts)
  return "Reponse : \n"+reponse

def web_explorer(input_user,title):
    urls = trouver_urls(input_user)
    if urls:
        liens = urls
        printdebug("liens : "+str(liens))
        reponse = web_qa(liens,input_user)
    else:
        liens = create_url(input_user)
        printdebug("liens : "+str(liens))
        reponse = search_on_google(input_user,liens)
    return ("TITRE : "+str(title)+"\n\nREPONSE : "+reponse+"\n\nSource : "+str(liens)+"\n")