# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

class AskOpenAI():
    @staticmethod
    def Ask(question) -> Text:
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Descrie succint: " + question}],
            max_tokens=250
        )["choices"][0]["message"]["content"]


class ActionUtterAirwayDescription(Action):

    def name(self) -> Text:
        return "action_utter_airway_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("obstructia cailor aeriene")

        buttons = [
            {"title": "Da", "payload": "am caile respiratorii afectate"},
            {"title": "Nu", "payload": "respir bine"},
            {"title": "Vreau explicatii", "payload": "vreau explicatii despre caile respiratorii"}
        ]
        dispatcher.utter_message(text=f"O scurta definitie: {description}<br><br>Aveti probleme cu caile respiratorii?", buttons=buttons)

        return []

class ActionUtterHemmorageDescription(Action):

    def name(self) -> Text:
        return "action_utter_hemmorage_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("hemoragie")

        dispatcher.utter_message(text=f"O scurta definitie: {description}<br><br>Aveti hemoragie?")

        return []
    
class ActionUtterSeizingDescription(Action):

    def name(self) -> Text:
        return "action_utter_seizing_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("convulsii")

        buttons = [
            {"title": "Da", "payload": "sufar de convulsii"},
            {"title": "Nu", "payload": "nu, nu sufar de convulsii"}
        ]
        dispatcher.utter_message(text=f"O scurta definitie: {description}<br><br>Aveti convulsii?", buttons=buttons)

        return []
    
class ActionUtterPainHemmorageDescription(Action):

    def name(self) -> Text:
        return "action_utter_pain_hemmorage_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("scara medicala a durerii de la 1 la 10")

        dispatcher.utter_message(text=f"O scurta definitie: {description}<br><br>Cum evaluati durerea pe o scara de la 1 la 10?")

        return []
    
class ActionUtterPainScaleDescription(Action):

    def name(self) -> Text:
        return "action_utter_pain_scale_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("scara medicala a durerii de la 1 la 10")

        buttons = [
            {"title": "Severa (7-10)", "payload": "durere severa"},
            {"title": "Moderata (4-6)", "payload": "durere moderata"},
            {"title": "Usoara (1-3)", "payload": "durere usoara"},
            {"title": "Nicio durere", "payload": "nicio durere"}
        ]
        dispatcher.utter_message(text=f"O scurta definitie: {description}<br><br>Cum evaluati durerea pe o scara de la 1 la 10?", buttons=buttons)

        return []
    
class ActionUtterGCSDescription(Action):

    def name(self) -> Text:
        return "action_utter_gcs_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("GCS - Glasgow Coma Scale")
        
        buttons = [
            {"title": "Sub 12 (inclusiv)", "payload": "GCS mic"},
            {"title": "13 -14", "payload": "GCS mediu"},
            {"title": "15", "payload": "GCS mare"}
        ]
        dispatcher.utter_message(text=f"O scurta definitie: {description}<br><br>Cum evaluati nivelul de constienta?", buttons=buttons)

        return []
    
class ActionUtterOxygenSaturationDescription(Action):

    def name(self) -> Text:
        return "action_utter_oxygen_saturation_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("saturatia de oxigen in sange")
        
        buttons = [
            {"title": "Sub 90%", "payload": "saturatie oxigen mica"},
            {"title": "90 - 92%", "payload": "saturatie oxigen medie"},
            {"title": "Peste 92%", "payload": "saturatie oxigen mare"}
        ]
        dispatcher.utter_message(text=f"O scurta definitie: {description}<br><br>Cum evaluati saturatia de oxigen in sange?", buttons=buttons)

        return []
    
class ActionUtterCompensatedShockOrangeDescription(Action):

    def name(self) -> Text:
        return "action_compensated_shock_orange"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        SlotSet("pain_high", True)
        
        # buttons = [
        #     {"title": "> 2 secunde", "payload": "timp CRT mic"},
        #     {"title": "<= 2 secunde", "payload": "timp CRT normal"}
        # ]
        # dispatcher.utter_message(text=f"Care este timpul CRT?", buttons=buttons)
        buttons = [
            {"title": "Da", "payload": "da, portocaliu am soc compensat pe portocaliu"},
            {"title": "Nu", "payload": "nu, nu portocaliu am soc compensat pe portocaliu"},
            {"title": "Vreau explicatii", "payload": "vreau explicatii portocaliu soc compensat portocaliu"}
        ]
        dispatcher.utter_message(text=f"Ati suferit soc compensat?", buttons=buttons)


        return []
    
class ActionUtterCompensatedShockYellowDescription(Action):

    def name(self) -> Text:
        return "action_compensated_shock_yellow"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        SlotSet("pain_high", False)
        
        # buttons = [
        #     {"title": "> 2 secunde", "payload": "timp CRT mic"},
        #     {"title": "<= 2 secunde", "payload": "timp CRT normal"}
        # ]
        # dispatcher.utter_message(text=f"Care este timpul CRT?", buttons=buttons)
        buttons = [
            {"title": "Da", "payload": "da, am suferit soc compensat pe galben"},
            {"title": "Nu", "payload": "nu, nu am suferit soc compensat pe galben"},
            {"title": "Vreau explicatii", "payload": "vreau explicatii soc compensat galben"}
        ]
        dispatcher.utter_message(text=f"Ati suferit soc compensat?", buttons=buttons)

        return []
    
class ActionUtterCompensatedShockOrangeDescription(Action):

    def name(self) -> Text:
        return "action_utter_compensated_shock_orange_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("termenul medical soc compensat")
        
        buttons = [
            {"title": "Da", "payload": "da, portocaliu am soc compensat pe portocaliu"},
            {"title": "Nu", "payload": "nu, nu portocaliu am soc compensat pe portocaliu"}
        ]
        dispatcher.utter_message(text=f"Ati suferit soc compensat?", buttons=buttons)

        return []
    
class ActionUtterCompensatedShockYellowDescription(Action):

    def name(self) -> Text:
        return "action_utter_compensated_shock_yellow_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("termenul medical soc compensat")
        
        buttons = [
            {"title": "Da", "payload": "da, am suferit soc compensat pe galben"},
            {"title": "Nu", "payload": "nu, nu am suferit soc compensat pe galben"}
        ]
        dispatcher.utter_message(text=f"Ati suferit soc compensat?", buttons=buttons)

        return []
    
class ActionUtterCRTNoPainDescription(Action):

    def name(self) -> Text:
        return "action_crt_no_pain_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("termenul medical soc compensat")
        
        buttons = [
            {"title": "> 2 secunde", "payload": "timp CRT mic fara durere"},
            {"title": "<= 2 secunde", "payload": "timp CRT normal fara durere"},
            {"title": "Nu stiu", "payload": "timp CRT normal fara durere"}
        ]
        dispatcher.utter_message(text=f"O scurta definitie: {description}<br><br>Care este timpul CRT?", buttons=buttons)

        return []
    
class ActionUtterCRTNoShockOrangeDescription(Action):

    def name(self) -> Text:
        return "action_utter_crt_no_shock_orange_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("termenul medical soc compensat")
        
        buttons = [
            {"title": "> 2 secunde", "payload": "fara soc portocaliu timp crt mare mare mare cod portocaliu"},
            {"title": "<= 2 secunde", "payload": "fara soc portocaliu timp crt normal cod portocaliu"},
            {"title": "Nu stiu", "payload": "fara soc portocaliu timp crt normal cod portocaliu"}
        ]
        dispatcher.utter_message(text=f"O scurta definitie: {description}<br><br>Care este timpul CRT?", buttons=buttons)

        return []
    
class ActionUtterCRTNoShockYellowDescription(Action):

    def name(self) -> Text:
        return "action_utter_crt_no_shock_yellow_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("termenul medical soc compensat")
        
        buttons = [
            {"title": "> 2 secunde", "payload": "cod galben timp crt mare mare mare cod galben"},
            {"title": "<= 2 secunde", "payload": "galben fara soc timp crt normal cod galben"},
            {"title": "Nu stiu", "payload": "galben fara soc timp crt normal cod galben"}
        ]
        dispatcher.utter_message(text=f"O scurta definitie: {description}<br><br>Care este timpul CRT?", buttons=buttons)

        return []
    
class ActionUtterDefault(Action):

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        description = AskOpenAI.Ask("termenul medical soc compensat")
        
        dispatcher.utter_message(text="Nu am inteles raspunsul. Va rog sa raspundeti mai clar.")

        return []