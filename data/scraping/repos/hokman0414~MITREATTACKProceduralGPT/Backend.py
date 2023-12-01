import pickle
import requests
import keyring
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
import json
import openai


# Keyring ChatGPT API token storage
service_id = 'Working'
user_name = 'ChatGPTAI'

#OpenAI API key
retrieved_key = keyring.get_password(service_id, user_name)

#langchain set up requirement to set up key like this
os.environ[f"OPENAI_API_KEY"] = retrieved_key

#calling LLM model
llm=ChatOpenAI(temperature=0.05,model_name='gpt-3.5-turbo-16k')
#==================================================Load the mitre Techniques JSON data from the file===============================================
input_filename = "mitre_techniques.json"
with open(input_filename, "r") as json_file:
    techniques_data = json.load(json_file)


#this is to query to T code entered from the Streamlit application
def queryTcode(tcode):
    # Input a Mitre T code
    input_t_code = tcode
    # Search for the provided T code in the data
    matching_technique = None
    for technique in techniques_data.get("techniques", []):
        if technique["Tcode"] == input_t_code:
            matching_technique = technique
            break


    description = matching_technique['description']
    Tname = matching_technique['name']
    Tcode = matching_technique['Tcode']
    # Define the prompt -> set to singular question for testing
    prompt_text = f'''generate a detailed question based on description below that can answer the Procedural level of the TTPs. 
                    output must:contain only the question and nothing else. don't contain T code or mitre. must include the name of the attack: {Tname}. No mentions of TTPs or procedural leve.
                    output must be: "how did the threat actor/malware specifically use" 
                    Mitre: {Tcode}
                    description:{description}'''

    output = llm.predict(prompt_text)
    print(output)
    return output


# Getting Dection rule being written
def DetectRuleCreate(llm,RuleType,tcode,procedure):
    # Hv to specificy Detection rule type such as KQL,Splunk, Elastic Search, YARA, SIGMA
    # Ex Detction Ruling=KQL
    DetectionRuling = str(RuleType)
    # Search for the provided T code in the data
    matching_technique = None
    for technique in techniques_data.get("techniques", []):
        if technique["Tcode"] == tcode:
            matching_technique = technique
            break
    if matching_technique is None:
        return "Technique not found"
    mitigation=matching_technique['mitigation']
    description = matching_technique['description']
    Tname = matching_technique['name']
    Tcode = matching_technique['Tcode']
    ptext=f'''write me {DetectionRuling} detection rule based on the procedure. Here all the necessary data. Output must be accurate to the MITRE Technique presented. only give me the rule in the detection rule format and nothing else included.
    Mitre T code:{Tcode}
    T code name: {Tname}
    Description: {description}
    Mitigations:{mitigation}
    Procedure of the actor: {procedure}
'''
    output = llm.predict(ptext)
    #print(output)
    return output



#==================================================================Set Attributes======================================

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"


#--------------------------------------------- LANGCHAIN HTML PARSING LOGIC -----------------------------------------
#call langchain variables
#what langchain.document_loaders do -> Document Loaders are responsible for loading documents into the LangChain system. They handle various types of documents, including PDFs, and convert them into a format that can be processed by the LangChain system.

#text splitter splits the text to smaller bit -> will need to tune seperator in future for errors
def get_text_chunks_langchain(text,link,title):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=13000, chunk_overlap=4400, separators=["\\n"])
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t, metadata={'source': link , 'title': title}) for t in texts]
    return docs

#PFIx this later
#getting the mitre TTPs when u put in the T code. use emdedding to extract the question via another QA retrieval
def QARetrieval(llm, VectorStore,tcode):
    for technique in techniques_data.get("techniques", []):
        if technique["Tcode"] == tcode:
            matching_technique = technique
            break
    if matching_technique is None:
        return "Technique not found"
    Tname = matching_technique['name']
    # chatGPT API Prompt template
    template=f'''{queryTcode(tcode)} output requirement: 1.put into a paragraph which bullet point specific technical commands,codes,or indicators that is related to the question and remove anything not relating to {Tname} and must only start with *threat group name or malware name only *used/exploited/orestablished*'''
    answer = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=VectorStore.as_retriever(search_type="mmr"))
    chain = answer.run(template)
    return chain

