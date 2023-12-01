import streamlit as st
from langchain.llms import OpenAI, OpenAIChat
from langchain.prompts import PromptTemplate, ChatPromptTemplate, BasePromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain, LLMBashChain
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import SimpleMemory, ConversationBufferMemory, ChatMessageHistory
from langchain.agents import AgentExecutor
import requests
import json
from simple_salesforce import Salesforce as sfSimple
import pandas as pd
from langchain.chains.llm_bash.prompt import BashOutputParser


class SalesforcePreAssistant:
    def __init__(self):
        self.llm = OpenAI(temperature=0.0, openai_api_key=st.secrets.openai.OPENAI_API_KEY)
        self.url_getid = "https://prod-24.westus.logic.azure.com:443/workflows/a236078c6312479abc2220c90063998c/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=swgcCo96aTVrTvm1eZR_GzC9kernAH_0Pkshyo2wICg"
        self.sfUsername = st.secrets.salesforce.SALESFORCE_USERNAME
        self.sfPassword = st.secrets.salesforce.SALESFORCE_PASSWORD
        self.sfToken = st.secrets.salesforce.SALESFORCE_TOKEN
        self.object_list = ['Account', 'User', 'Order', 'Opportunity', 'Lead', 'Note', 'Consumable__c', 'Case']
        self.memory = ConversationBufferMemory()

    def getnameids(self, varName, varType):
        sf = sfSimple(username=self.sfUsername, password=self.sfPassword, security_token=self.sfToken)
        varURL = self.url_getid
        body = {"search_object": varType, "search_value": varName}

        response = requests.post(varURL, json=body)
        return response.json()

    def getfields(self, varObject):
        sf = sfSimple(username=self.sfUsername, password=self.sfPassword, security_token=self.sfToken)
        sdescribe = getattr(sf, varObject).describe()
        sfields = sdescribe['fields']
        sfieldnames = []
        for field in sfields:
            sfieldnames.append(field['name'])
        return sfieldnames

    def process_input(self, input_string):
        names_prompt = PromptTemplate(
            input_variables=["userinput"],
            template=("""
                Identify the named entities from the users request: {userinput}. 
                Categorize them as a User or Account (these are the only two values).
                There should not be any other types other than User or Account.  
                Return only a json object for each named entity in the following format: search_object: object value, search_value: name value.
                Place each json object into a single array with just the array.
                      
                Review your answer - if you have any other categorization other than Account or User you need to change it. 
                """
            )
        )

        names_chain = LLMChain(llm=self.llm, prompt=names_prompt)
        namelist = names_chain.run(input_string)
        namelist = json.loads(namelist)
        responselist = []

        for nameitem in namelist:
            searchobject=nameitem['search_object']
            searchvalue=nameitem['search_value']
            response = self.getnameids(searchvalue, searchobject)
            if 'error' not in response:
                responselist.append(response)
        return responselist
    
    def process_object(self, userprompt):
        fields_prompt = PromptTemplate(
            input_variables=['object_list','user_input'],
            template=("""
                You are a programming expert. You specialize in salesforce.
                You will identify the primary object mentioned in the user request. 
                The primary object will be the object to be created, updated, or to get information about.
                Respond only with the value of the object - one word corresponding to the object. No other commentary or words should be provided. 
                Objects will be one of the following: {object_list}

                User Input: {user_input}
                """
            )
        )

        fields_chain = LLMChain(llm=self.llm, prompt = fields_prompt)
        fields_chain_output = fields_chain.run({"object_list": self.object_list, "user_input": userprompt})
        fields_chain_output = fields_chain_output.split()[-1]                                                   
        fields_list=self.getfields(fields_chain_output)

        return fields_list


    def get_SalesforcePreAssistant(self, varUserInput):
        response_getids = self.process_input(varUserInput)
        response_getfields = self.process_object(varUserInput)
        prompt = PromptTemplate(
            input_variables=["varUserInput", "response_getids", "response_getfields"],
            template=("""
                You are a programming expert and helpful assistant. 
                You will create bash or python code using simple_salesforce based on the request of the user. 
                You will be given a list of relevant Ids and fields to help construct this code. 
                Id fields should use the value in recordid. Ex: Id, OwnerId, AccountId, etc.. should use the recordid provided.
                Do not add any fields that are not directly mentioned or implicitly inferred in the users input.    
                Return only the code.
                    User Request: {varUserInput}
                    Relevant Ids: {response_getids}
                    Relevant Fields: {response_getfields}
                    """
            )
        )
        schain = LLMChain(llm=self.llm, prompt=prompt)
        sresponse = schain.run({"varUserInput": varUserInput, "response_getids": response_getids, "response_getfields": response_getfields})
        return sresponse
