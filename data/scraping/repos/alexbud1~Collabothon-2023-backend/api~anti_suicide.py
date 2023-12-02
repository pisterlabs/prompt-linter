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

class Anti_Suicide_Bot:
    def __init__(self):
        self.conversation_history_agent = "YOU ARE DEALING WITH A SENSITIVE KID THAT WAS TRYING TO COMMIT SUICIDE, BE GENTLE, TRY TO CALM HIM DOWN  INPUT: {user_input} . OUTPUT:"
        
        GenParams().get_example_values()
        
        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 50,
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
        anti_suicade_response= self.llm(prompt)['text']

        return anti_suicade_response
if __name__ == "__main__":
    anti_suicide_bot = Anti_Suicide_Bot()
    print("Jestem tu dla Ciebie i chcę cię wysłuchać. Czy jest coś co mogę dla ciebie zrobić?: ")
    while True:
        user_message = input("ty: ")
        if user_message.lower() in ['exit', 'quit']:
            break
        response = anti_suicide_bot.get_response(user_message)
        print("słoń:", response)
