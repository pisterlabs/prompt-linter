import json
from utils import prettify
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from metadata import Metadata
import logging


llm = OpenAI(temperature=0)
apis = ["https://us10.graph.sap/api/demo/my.custom/$metadata", "https://us10.graph.sap/api/demo/company/$metadata"]

apiMetadata = ""

class Template:
   
    def prepareTemplate():
        templatePart1 = """You are a helpful assistant who reads the edmx odata metadata and then prepares an odata query based on that. Read the given odata metadata in edmx format and their respective URLs carefully for many or one APIs
        Details about the APIs:
        """

        templatePart2 = ""
        templatePart2ForLogging = ""

        templatePart3 = """Read the question below carefully and decide which API would be suitable for the question below. The response should be the correct URL with the odata v4 filters.
        
        Rules: 
        1. Only use $expand for fields with type NavigationProperties and which are of type Collection
        for example : 
        2. DO NOT use $expand for fields with type Properties which are ComplexType
        3. You can use simple $select for fields with type Properties
        4. Use $select in $expand for fields with type NavigationProperties
        
        For example consider the following : 
        <Property Name="id" Type="Edm.String" Nullable="false"/>
        <Property Name="name" Type="Edm.String"/>
        <NavigationProperty Name="xxxx" Type="Collection(my.url.someCollection)" Partner="up_" ContainsTarget="true"/>
        <Property Name="bbbb" Type="my.url.someComplexTypeButNotCollection"/>
        <NavigationProperty Name="aaaa" Type="my.url.someOtherCollection"/>
        
        You must use $expand for NavigationProperty with name "xxxx" and with name "aaaa", and not for any other fields as they are not NavigationProperties. To use select in the NavigationProperties, use it in the $expand block instead. You must not use $expand for properties with names "bbbb", "id" and "name" as they are properties and not NavigationProperties

        ONLY print the url and nothing more. If the question is not relevant to the metadata return 'Not found'. The url is created using the entity name"""
        metadata = Metadata()

        for index,url in enumerate(apis): 
            metadataString = metadata.fetchApiMetadata(url)
            templatePart2 = templatePart2 + "API %(index)s (URL: %(url)s) \n %(metadataString)s \n" %{"index": index+1, "url":url,  "metadataString": metadataString }
            templatePart2ForLogging = templatePart2ForLogging + "\nAPI %(index)s (URL: %(url)s) \n <xml> ... </xml> \n" %{"index": index+1, "url":url }
        logging.info("Query Sent to LLM: \n\n" + templatePart1 + templatePart2ForLogging + templatePart3)
        return templatePart1 + templatePart2 + templatePart3
    
    def prepareTemplateForProcessingJsonResponse(jsonData):
        templatePart1 = "You are a helpful assistant who reads the json response from an http odata rest query. Read the JSON below \n"
        templatePart3 = "\nRespond with a human readable response to the question below based on the json above."
        logging.info("Query Sent to LLM: \n\n" + templatePart1 + prettify(jsonData) + templatePart3)
        return templatePart1 + json.dumps(jsonData).replace('{', '{{').replace('}','}}') + templatePart3
