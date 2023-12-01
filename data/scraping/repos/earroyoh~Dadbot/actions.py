from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction
from rasa_sdk.events import (
    SlotSet,
    UserUtteranceReverted,
    ConversationPaused,
    EventType,
    FollowupAction,
)

import requests
import html2text
import json
import random
import logging
import os
import openai

logger = logging.getLogger(__name__)

class WikipediaAction(Action):
    def name(self):
        return "action_wikipedia_pregunta"

    def run(self, dispatcher, tracker, domain):

        intent = tracker.latest_message["intent"].get("name")
        logger.info(intent)
        try:
            entity = tracker.latest_message["entities"][0]["entity"] 
        except:
            entity = "cosa"
        logger.info(entity)
        value = next(tracker.get_latest_entity_values(entity), None)
        logger.info(value)

        if value != None:
            pregunta = value

            r = requests.get("https://es.wikipedia.org/w/api.php?action=query&list=search&srprop=snippet&format=json&origin=*&utf8=&srsearch={}".format(pregunta))
            response = r.json()
            response = response["query"]["search"][0]["snippet"]
            response = response.replace('<span class=\"searchmatch\">',"").replace('</span>',"")
            response = response.split('.')[0] + "."
                
            dispatcher.utter_message(text="Dice la Wikipedia: " + format(response))

        else:
            flip = random.random()
            if flip > 0.5: 
                dispatcher.utter_message(text="No entiendo lo que me dices")
            else:
                dispatcher.utter_message(text="No sé lo que me quieres decir")
        return []
    
class WeatherAction(Action):
    def name(self):
        return "action_openweather_tiempo"
     
    def run(self, dispatcher, tracker, domain):
        ciudad = tracker.get_slot("ciudad")
        if (format(ciudad) == "None"): 
             ciudad = "Getafe" ## Initialization
             text = "Aquí en Getafe hay "
        else:
             text = "En " + ciudad + " hay "
        
        # OpenWeatherMap API - Pseudonimo API key
        query = ciudad + ',es&lang=es&units=metric&appid=52b049e3be4e6efd8cff05a01210b266'
        r = requests.get('https://api.openweathermap.org/data/2.5/weather?q={}'.format(query))
        response = r.json()
        #print(response)
        cielo = response["weather"][0]["description"]
        #print(cielo)
        temperatura = int(response["main"]["temp"])
        #print(temperatura)
        
        dispatcher.utter_message(text = text + format(cielo) + " y una temperatura de " + format(temperatura) + " grados")
        return []

class NewsAction(Action):
    def name(self):
        return "action_ultimas_noticias"

    def run(self, dispatcher, tracker, domain):

        # RTVE JSON latest news
        r = requests.get("http://www.rtve.es/api/noticias.json")
        response = r.json()
        response = response["page"]["items"][0]["longTitle"]
        response = html2text.html2text(response).replace('*',"").replace('\n', "")
        response = response.split('.')[0] + "."

        dispatcher.utter_message(text="Estas son las últimas noticias: " + format(response))
        return []

class OpenAI_QA(Action):
    def name(self):
        return "action_openai_qa"

    def run(self, dispatcher, tracker, domain):
        stop = "\n"

        prompt = """Q: ¿Cual es la esperanza de vida en Estados Unidos?
        A: La esperanza de vida en Estados Unidos es de 78 años.

        Q: ¿Quién fue presidente de España en 1982? 
        A: Felipe González fue presidente de España en 1982.

        Q: ¿A qué partido pertenecía?
        A: Pertenecía al Partido Socialista Obrero Español.

        Q: ¿Quién fue presidente después de José María Aznar?
        A: Mariano Rajoy fue presidente después de José María Aznar.

        Q: ¿Qué equipo ganó La Liga en 2010?
        A: En 2010 el Fútbol Club Barcelona ganó La Liga.

        Q:""" + tracker.latest_message["text"] + """ 
        A:"""

        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai_response = openai.Completion.create(engine="davinci", max_tokens=50, prompt=prompt, stop=stop, temperature=0)
        response = openai_response["choices"][0]["text"]

        dispatcher.utter_message(text=format(response))
        return [SlotSet("GPT3", "true")]

class OpenAI_chat(Action):
    def name(self):
        return "action_openai_chat"

    def run(self, dispatcher, tracker, domain):
        stop = "\nHumano: IA:"

        prompt="""Humano: Hola, ¿te conozco?
        IA: Soy una IA creada por OpenAI. ¿De qué quieres hablar hoy?

        Humano: """ + tracker.latest_message["text"] + """
        IA:"""

        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai_response = openai.Completion.create(engine="davinci", max_tokens=150, prompt=prompt, stop=stop, temperature=0.4, top_p=1, frequency_penalty=0.0, presence_penalty=0.6)
        response = openai_response["choices"][0]["text"].replace(" Humano:","\n").split("\n")[0]

        dispatcher.utter_message(text=format(response))
        return [SlotSet("GPT3", "true")]
