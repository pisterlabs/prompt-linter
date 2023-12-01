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

        prompt = """Q: What is human life expectancy in the United States?
        A: Human life expectancy in the United States is 78 years.

        Q: Who was president of the United States in 1955?
        A: Dwight D. Eisenhower was president of the United States in 1955.

        Q: What party did he belong to?
        A: He belonged to the Republican Party.

        Q: Who was president of the United States before George W. Bush?
        A: Bill Clinton was president of the United States before George W. Bush.

        Q: Who won the World Series in 1995?
        A: The Atlanta Braves won the World Series in 1995.

        Q: What year was the first fax sent?
        A:"""

        response = openai.Completion.create(model="davinci", prompt=prompt, stop=stop, temperature=0)

        dispatcher.utter_message(text=format(response))
        return []
