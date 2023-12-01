#DO NOT PLAY WITH THE PROMPTS 

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
import os
from os.path import dirname, join 
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)
API_TOKEN = os.environ.get("APIKEY")
PROJECT_ID = os.environ.get("PROJECT_ID")


class Agent_Bot:
    def __init__(self):
        self.conversation_history_agent = "READ USER INPUT AND DECIDE IF THE USER IS ANGRY OUTPUT=ANGRY IF FEELS HAPPY OUTPUT=HAPPY IF SAD OUTPUT=SAD,  INPUT: {user_input} . OUTPUT:"
        
        GenParams().get_example_values()
        
        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 4,
            GenParams.TEMPERATURE: 0,
            GenParams.REPETITION_PENALTY: 1,
        }

        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": "JRaJbPGNBfPBrE69CCnHupv-7ON_Zf82qZEmGuvaCtfB",  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id="f2724f63-3bef-4fdc-a2be-123226e856a5"
        )
        self.llm = LLMChain(llm=self.model.to_langchain(), prompt=PromptTemplate.from_template(self.conversation_history_agent))

    def get_response(self, user_input):
        prompt = user_input
        agent_response = self.llm(prompt)['text']

        return agent_response


class Sad_Bot:
    def __init__(self):
        self.conversation_history_sad = "ACT LIKE A ONLINE PSYCHOLOGIST FOR CHILDREN, BE KIND AND NICE. LISTEN TO THEM, SUPPORT AND HELP THEM TO FEEL BETTER. INPUT: {user_input} . OUTPUT:"
        
        GenParams().get_example_values()
        
        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 140,
            GenParams.TEMPERATURE: 0,
            GenParams.REPETITION_PENALTY: 1,
        }

        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": "JRaJbPGNBfPBrE69CCnHupv-7ON_Zf82qZEmGuvaCtfB",  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id="f2724f63-3bef-4fdc-a2be-123226e856a5"
        )
        self.llm = LLMChain(llm=self.model.to_langchain(), prompt=PromptTemplate.from_template(self.conversation_history_sad))

    def get_response(self, user_input):
        prompt = user_input
        sad_bot_response = self.llm(prompt)['text']

        return sad_bot_response

class Happy_Bot:
     def __init__(self):
        self.conversation_history_happy = "ACT LIKE ONLINE PSYCHOLOGIST YOU ARE HAPPY THAT THE CHILDREN IS HAPPY, TALK WITH CHILD, APPRECIATE IT BE PROUD OF THEM , BE KIND AND NICE. LISTEN TO THEM, SUPPORT THEM INPUT: {user_input} . OUTPUT:"


        GenParams().get_example_values()
        
        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 140,
            GenParams.TEMPERATURE: 0.5,
            GenParams.REPETITION_PENALTY: 1,
        }

        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": "JRaJbPGNBfPBrE69CCnHupv-7ON_Zf82qZEmGuvaCtfB",  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id="f2724f63-3bef-4fdc-a2be-123226e856a5"
        )
        self.llm = LLMChain(llm=self.model.to_langchain(), prompt=PromptTemplate.from_template(self.conversation_history_happy))

     def get_response(self, user_input):
        prompt = user_input
        happy_bot_response = self.llm(prompt)['text']

        return happy_bot_response
     
class Angry_Bot:
      def __init__(self):
        self.conversation_history_angry = "ACT LIKE ONLINE PSYCHOLOGIST, YOU ARE WORKING WITH AN ANGRY CHILDREN, TRY TO CALM HIM, LISTEN TO THEM, PROPOSE SOME TIPS HOW TO CALM DOWN, SUPPORT THEM INPUT: {user_input} . OUTPUT:"


        GenParams().get_example_values()
        
        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 140,
            GenParams.TEMPERATURE: 0.3,
            GenParams.REPETITION_PENALTY: 1,
        }

        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": "JRaJbPGNBfPBrE69CCnHupv-7ON_Zf82qZEmGuvaCtfB",  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id="f2724f63-3bef-4fdc-a2be-123226e856a5"
        )
        self.llm = LLMChain(llm=self.model.to_langchain(), prompt=PromptTemplate.from_template(self.conversation_history_angry))

      def get_response(self, user_input):
        prompt = user_input
        angry_bot_response = self.llm(prompt)['text']

        return angry_bot_response   
      


    
class Friendly_Bot:
      def __init__(self):
        self.conversation_history_friendly = "ACT LIKE A GOOD BUDDY TO A CHILDREN, TALK ABOUT THEIR DAY, ABOUT THINGS THAT THEY LIKE, BE KIND, LISTEN TO THEM, SUPPORT THEM. INPUT: {user_input} . OUTPUT:"


        GenParams().get_example_values()
        
        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 140,
            GenParams.TEMPERATURE: 0.3,
            GenParams.REPETITION_PENALTY: 1,
        }

        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": "JRaJbPGNBfPBrE69CCnHupv-7ON_Zf82qZEmGuvaCtfB",  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id="f2724f63-3bef-4fdc-a2be-123226e856a5"
        )
        self.llm = LLMChain(llm=self.model.to_langchain(), prompt=PromptTemplate.from_template(self.conversation_history_friendly))

      def get_response(self, user_input):
        prompt = user_input
        friendly_bot_response = self.llm(prompt)['text']

        return friendly_bot_response   
    

if __name__ == "__main__":
    agent_bot = Agent_Bot()
    sad_bot = Sad_Bot()
    happy_bot = Happy_Bot()
    angry_bot = Angry_Bot()
    frienfly_bot = Friendly_Bot()


    first_message = input("Hej! Jestem tutaj po to by ci pomóc ;) powiedz mi jak się miewasz?: ")
    response = agent_bot.get_response(first_message)
    print(response)
    response_cleaned = response.strip().upper()



    if "HAPPY" in response.upper():
      response = happy_bot.get_response(first_message)
      print("response:", response)
      while True:
            user_message = input("Ty: ")
            if user_message.lower() in ['exit', 'quit']:
                break 
            response = happy_bot.get_response(user_message)
            print("response:", response)
    elif "ANGRY" in response.upper():
       response = angry_bot.get_response(first_message)
       print("response:", response) 
       while True:
            user_message = input("Ty: ")
            if user_message.lower() in ['exit', 'quit']:
                break 
            response = angry_bot.get_response(user_message)
            print("response:", response)        
    elif "SAD" in response.upper():
       response = sad_bot.get_response(first_message)
       print("response:", response)
       while True:
            user_message = input("Ty: ")
            if user_message.lower() in ['exit', 'quit']:
                break 
            response = sad_bot.get_response(user_message)
            print("response:", response)
    else:
       response = frienfly_bot.get_response(first_message)
       print("response:", response)
       while True:
            user_message = input("Ty: ")
            if user_message.lower() in ['exit', 'quit']:
                break 
            response = frienfly_bot.get_response(user_message)
            print("response:", response)
    
    
