import os

from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

"""Defining utility functions for constructing a readable exchange
"""

def system_output(output):
    """Function for printing out to the user
    """
    # print('======= Bot =======')
    print(output)

def user_input():
    """Function for getting user input
    """
    # print('======= Human Input =======')
    return input('input: ')

def parsing_info(output):
    """Function for printing out key info
    """
    print(f'*Info* {output}')

"""Implimenting the conversation as a directed graph
"""


from typing import List

class Edge:

    """Edge
    at it's highest level, an edge checks if an input is good, then parses
    data out of that input if it is good
    """

    def __init__(self, condition, parse_prompt, parse_class, llm, max_retrys=3, out_node=None):
        """
        condition (str): a True/False question about the input
        parse_query (str): what the parser whould be extracting
        parse_class (Pydantic BaseModel): the structure of the parse
        llm (LangChain LLM): the large language model being used
        """
        self.condition = condition
        self.parse_prompt = parse_prompt
        self.parse_class = parse_class
        self.llm = llm

        #how many times the edge has failed, for any reason, for deciding to skip
        #when successful this resets to 0 for posterity.
        self.num_fails = 0

        #how many retrys are acceptable
        self.max_retrys = max_retrys

        #the node the edge directs towards
        self.out_node = out_node

    def check(self, _input):
        """ask the llm if the input satisfies the condition
        """
        validation_query = f'following the output schema, does the input satisfy the condition?\ninput:{_input}\ncondition:{self.condition}'
        class Validation(BaseModel):
            is_valid: bool = Field(description="if the condition is satisfied")
        parser = PydanticOutputParser(pydantic_object=Validation)
        _input = f"Answer the user query.\n{parser.get_format_instructions()}\n{validation_query}\n"
        return parser.parse(self.llm(_input)).is_valid

    def parse(self, _input):
        """ask the llm to parse the parse_class, based on the parse_prompt, from the input
        """
        parse_query = f'{self.parse_prompt}:\n\n"{_input}"'
        parser = PydanticOutputParser(pydantic_object=self.parse_class)
        _input = f"Answer the user query.\n{parser.get_format_instructions()}\n{parse_query}\n"
        return parser.parse(self.llm(_input))


    def execute(self, _input):
        """Executes the entire edge
        returns a dictionary:
        {
            continue: bool,       weather or not should continue to next
            result: parse_class,  the parsed result, if applicable
            num_fails: int        the number of failed attempts
            continue_to: Node     the Node the edge continues to
        }
        """

        #input did't make it past the input condition for the edge
        if not self.check(_input):
            self.num_fails += 1
            if self.num_fails >= self.max_retrys:
                return {'continue': True, 'result': None, 'num_fails': self.num_fails, 'continue_to': self.out_node}
            return {'continue': False, 'result': None, 'num_fails': self.num_fails, 'continue_to': self.out_node}

        try:
            #attempting to parse
            self.num_fails = 0
            return {'continue': True, 'result': self.parse(_input), 'num_fails': self.num_fails, 'continue_to': self.out_node}
        except:
            #there was some error in parsing.
            #note, using the retry or correction parser here might be a good idea
            self.num_fails += 1
            if self.num_fails >= self.max_retrys:
                return {'continue': True, 'result': None, 'num_fails': self.num_fails, 'continue_to': self.out_node}
            return {'continue': False, 'result': None, 'num_fails': self.num_fails, 'continue_to': self.out_node}


class Node:

    """Node
    at it's highest level, a node asks a user for some input, and trys
    that input on all edges. It also manages and executes all
    the edges it contains
    """

    def __init__(self, prompt, retry_prompt):
        """
        prompt (str): what to ask the user
        retry_prompt (str): what to ask the user if all edges fail
        parse_class (Pydantic BaseModel): the structure of the parse
        llm (LangChain LLM): the large language model being used
        """

        self.prompt = prompt
        self.retry_prompt = retry_prompt
        self.edges = []

    def run_to_continue(self, _input):
        """Run all edges until one continues
        returns the result of the continuing edge, or None
        """
        for edge in self.edges:
            res = edge.execute(_input)
            if res['continue']: return res
        return None

    def execute(self):
        """Handles the current conversational state
        prompots the user, tries again, runs edges, etc.
        returns the result from an adge
        """

        #initial prompt for the conversational state
        system_output(self.prompt)

        while True:
            #getting users input
            _input = user_input()

            #running through edges
            res = self.run_to_continue(_input)

            if res is not None:
                #parse successful
                parsing_info(f'parse results: {res}')
                return res

            #unsuccessful, prompting retry
            system_output(self.retry_prompt)

    def ingest(self, input):
        """Handles the current conversational state
        prompots the user, tries again, runs edges, etc.
        returns the result from an adge
        """
        self.input = input
        #initial prompt for the conversational state
        system_output(self.prompt)

        while True:
            #getting users input

            #running through edges
            res = self.run_to_continue(self.input)

            if res is not None:
                #parse successful
                parsing_info(f'parse results: {res}')
                return res

            #unsuccessful, prompting retry
            system_output(self.retry_prompt)




# Defining Nodes
intencion_node = Node("Hola! soy Dana tu asistente,  en que puedo ayudarte ? ", "Disculpa, no te entiendo. Podrias repetir ?")
name_node = Node("Podrias decirme tu nombre?", "Disculpa, no te entendi. Podrias repetir tu nombre ?")
dni_node = Node("Podrias decirme tu DNI ?", "Disculpa, no entiendo. Necesitamos tu numero de documento para poder continuar. Podrias facilitarlo ?")
contact_node = Node("do you have a phone number or email we can use to contact you?", "I'm sorry, I didn't understand that. Can you please provide a valid email or phone number?")
budget_node = Node("What is your monthly budget for rent?", "I'm sorry, I don't understand the rent you provided. Try providing your rent in a format like '$1,300'")
avail_node = Node("Great, When is your soonest availability?", "I'm sorry, one more time, can you please provide a date you're willing to meet?")

#Defining Data Structures for Parsing
class saludoTemplate(BaseModel): output: str = Field(description="un saludo")
class nameTemplate(BaseModel): output: str = Field(description="a persons name")
class dniTemplate(BaseModel): output: str = Field(description="documento nacional de identidad de entre 7 u 8 numeros")
class phoneTemplate(BaseModel): output: str = Field(description="phone number")
class emailTemplate(BaseModel): output: str = Field(description="email address")
class budgetTemplate(BaseModel): output: float = Field(description="budget")
class dateTemplate(BaseModel): output: str = Field(description="date")

#defining the model
model_name = "text-davinci-003"
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

#Defining Edges
intencion_edge = Edge("El input contiene una frase", "Extrae la intencion en un maximo de dos palabras", saludoTemplate, model)
name_edge = Edge("Does the input contain a persons name?", " Extract the persons name from the following text.", nameTemplate, model)
dni_edge = Edge("El input contiene una frase con numeros o pueden ser solo numeros ", " En cualquier caso extrae los numeros del texto.", dniTemplate, model)
contact_phone_edge = Edge("does the input contain a valid argentinian phone number?", "extract the phone number in the format xxx-xxxx-xxxx", phoneTemplate, model)
contact_email_edge = Edge("does the input contain a valid email?", "extract the email from the following text", emailTemplate, model)
budget_edge = Edge("Does the input contain a number in the thousands?", "Extract the number from the following text from the following text. Remove any symbols and multiply a number followed by the letter 'k' to thousands.", budgetTemplate, model)
avail_edge = Edge("does the input contain a date or day? dates or relative terms like 'tommorrow' or 'in 2 days'.", "extract the day discussed in the following text as a date in mm/dd/yyyy format. Today is September 23rd 2023.", dateTemplate, model)

#Defining Node Connections
intencion_node.edges = [intencion_edge]
name_node.edges = [name_edge]
dni_node.edges = [dni_edge]
contact_node.edges = [contact_phone_edge, contact_email_edge]
budget_node.edges = [budget_edge]
avail_node.edges = [avail_edge]

#defining edge connections
intencion_edge.out_node = dni_node
dni_edge.out_node = name_node
name_edge.out_node = contact_node
contact_phone_edge.out_node = budget_node
contact_email_edge.out_node = budget_node
budget_edge.out_node = avail_node

#running the graph
current_node = intencion_node
while current_node is not None:
    res = current_node.execute()
    print(res)
    if res['continue']:
        current_node = res['continue_to']
