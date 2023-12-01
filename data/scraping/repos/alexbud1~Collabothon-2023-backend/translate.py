#DO NOT PLAY WITH THE PROMPTS FOR TRANSLATION 
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain import PromptTemplate
from langchain.chains import LLMChain
import os
from os.path import dirname, join 
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)
API_TOKEN = os.environ.get("APIKEY")
PROJECT_ID = os.environ.get("PROJECT_ID")

class ChatbotWithHistory:
    def __init__(self):
        self.conversation_history = "ACT LIKE A ONLINE PSYCHOLOGIST FOR CHILDREN, BE KIND AND NICE. LISTEN TO THEM, SUPPORT AND HELP THEM TO FEEL BETTER. INPUT: {user_input} . OUTPUT:"
        
        GenParams().get_example_values()
        
        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 50,
            GenParams.MAX_NEW_TOKENS: 200,
            GenParams.TEMPERATURE: 0,
            GenParams.REPETITION_PENALTY: 1,
        }

        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": API_TOKEN,  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id=PROJECT_ID
        )
        self.llm = LLMChain(llm=self.model.to_langchain(), prompt=PromptTemplate.from_template(self.conversation_history))

    def get_response(self, user_input):
        prompt = user_input
        # self.llm_chain.prompt += user_input
        bot_response = self.llm(prompt)['text']
        # self.llm_chain.prompt += bot_response

        return bot_response

#upon receiving a prompt
class Chatbot_translator_PL_to_EN:
    def __init__(self):
        self.conversation_history1 = 'TRANSLATE THE FOLLOWING INPUT FROM POLISH TO ENGLISHJ. RETURN JUST TRANSLATED MESSAGE. INPUT:  {human_input} OUTPUT: \n'
        GenParams().get_example_values()

        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 200,
            GenParams.TEMPERATURE: 0,
            GenParams.REPETITION_PENALTY: 1,
        }

        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": API_TOKEN,  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id= PROJECT_ID
        )
        self.llm = LLMChain(llm=self.model.to_langchain(), prompt=PromptTemplate.from_template(self.conversation_history1))

    def get_translation_ptoe(self, user_input:str):
        prompt = user_input
        translated_input = self.llm(prompt)['text']
        return translated_input

#upon receiving model output 
class Chatbot_translator_EN_to_PL:
    def __init__(self):
        self.conversation_history2 = 'TRANSLATE THE FOLLOWING INPUT FROM ENGLISH TO POLISH. RETURN JUST TRANSLATED MESSAGE. INPUT:  {human_input} OUTPUT: \n'
        GenParams().get_example_values()

        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 300,
            GenParams.TEMPERATURE: 0,
            GenParams.REPETITION_PENALTY: 1,
        }

        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": API_TOKEN,  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id=PROJECT_ID
        )
        self.llm = LLMChain(llm=self.model.to_langchain(), prompt=PromptTemplate.from_template(self.conversation_history2))

    def get_translation_etop(self, user_input:str):
        prompt = user_input
        translated_input1 = self.llm(prompt)['text']
        return translated_input1


# if __name__ == "__main__":
#     chatbot = ChatbotWithHistory()
#     translator1 = Chatbot_translator_PL_to_EN()
#     translator2 = Chatbot_translator_EN_to_PL()

#     while True:
#         user_message = input("ty: ")
#         trans_mess =translator1.get_translation1(user_message)
#         if user_message.lower() in ['exit', 'quit']:
#             break
#         # user_message1= translator.translate_text(user_message, target_lang="EN-US")
#         response = chatbot.get_response(trans_mess)
#         # import pdb
#         # pdb.set_trace()
#         trans_resp = translator2.get_translation2(response)

#         #response1 =translator.translate_text(response, target_lang="PL")
#         print("psycholog:", trans_resp)