import getpass
import os
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning import APIClient
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
import logging
import googletrans

logger = logging.getLogger()


class LlamaModel:
    def __init__(self, statement_to_classify):
        logger.debug("In __init__")
        self.credentials = {
            "url":  "https://eu-de.ml.cloud.ibm.com",
            "apikey": os.getenv("APIKEY", ""),
        }
        self.project_id = os.getenv("PROJECTID", "")
        self.model = self.create_model()
        self.statement = statement_to_classify
        self.statement_en = ""

    def create_model(self):
        logger.debug("In create_model")
        model_id_llama = ModelTypes.LLAMA_2_70B_CHAT
        parameters_llama = {
            GenParams.MAX_NEW_TOKENS: 1000,
        }
        flan_llama_model = Model(
            model_id=model_id_llama,
            credentials=self.credentials,
            project_id=self.project_id,
            params=parameters_llama,
        )
        return flan_llama_model

    def create_prompt_template(self):
        logger.debug("In create_prompt_template")
        # ocena_prompt = "Please Recognise positive sentiment in below statement: {input_variables}."
        # ocena_prompt = "Please Recognise emotions and their intensity in below statement: {input_variables}." \
        #                "Put it all in one json, with key as emotion you have " \
        #                "detected and value as its intensity"
        ocena_prompt = "Please make sentiment analysis in below statement that is written in polish: {input_variables}." \
                       "Recognise emotions and their intensity. Put it all in one json, with key as emotion you have" \
                       "detected and value as its intensity. Give me only this json."
        prompt_llama = PromptTemplate(
            input_variables=["statement"],
            template=ocena_prompt,
        )
        logger.debug(prompt_llama.dict())
        return prompt_llama

    def run_model(self):
        logger.debug("In run_model")
        if self.model is None:
            raise ValueError("Model is not created. Call create_model first.")
        logger.debug(self.statement)
        prompt_llama = self.create_prompt_template()
        logger.debug(f"in run_model, prompt_llama is: {prompt_llama.dict()}")
        flan_to_llama = LLMChain(llm=self.model.to_langchain(), prompt=prompt_llama, verbose=1)
        logger.debug(f"in run model, flan to llama is: {flan_to_llama}")
        qa = SimpleSequentialChain(chains=[flan_to_llama], verbose=True)
        logger.debug(f"in run_model, qa is : {qa}")
        outcome = qa.run(self.statement)
        # self.translate_to_en()
        # outcome = qa.run(self.statement_en)
        # outcome_pl = self.translate_to_pl(outcome)
        # logger.debug(outcome_pl)
        # return outcome_pl
        return outcome

    def translate_to_en(self):
        print(googletrans.LANGUAGES)
        translator = googletrans.Translator()
        result = translator.translate(self.statement, src='pl', dest='en')
        print(f"TRANSLACJA TO EN: {result.text}")
        self.statement_en = result.text

    def translate_to_pl(self, llm_result):
        print(googletrans.LANGUAGES)
        translator = googletrans.Translator()
        result = translator.translate(llm_result, src='en', dest='pl')
        print(f"TRANSLACJA TO PL: {result.text}")
        return result.text
