
import os
import openai
import requests


class Generator():
    """
    A Class that generate idea for user based GPT-3 API.
    """
    def __init__(self, prepared_question:str, number_of_idea: int, workshop_method:str, crazy:bool = False) -> None:
        """
        A constructor function for Generator class.
        :param prepared_question: A question that user want to ask.
        :param number_of_idea: Number of idea that user want to get.
        :param crazy: A boolean value that indicate if user want to get an unusual suggestions or a more normal one.
        :param workshop_method: A string value that indicate which workshop method user want to use.
        """
        self.prepared_question:str = prepared_question
        self.number_of_idea:int = number_of_idea
        self.crazy:bool = crazy
        self.workshop_method:str = workshop_method
        self.raw_result:dict = None
        self.idea_list:list = []
        self.idea_list_enhaced:list = []
        self.api_key:str = None
        self.payload:dict = {}

    def connect_openai(self) -> bool:
        """
        A function that create api connection.
        :return: A boolean value that indicate if api connection is created or not.
        """
        try:
            credential = open("data/api_credentials.json", "r")
            self.api_key = eval(credential.read())['openAI_api_key']
            print(self.api_key)
            openai.api_key = self.api_key
        except openai.exceptions.InvalidAPIKeyError:
            return False
        except FileNotFoundError:
            return False
        except openai.exceptions.InvalidRequestError:
            return False
        else:
            return True
        


    def generate_idea(self) -> bool:
        """
        A Funtion that generate idea for user based on GPT-3 API.
        :return: A boolean value that indicate if idea is generated or not.
        """
        if self.connect_openai():
            """
            If api connection is created, then generate idea.
            """
            # read parameters from txt file
            read_file = open("data/params_dict.txt", "r")
            params_dict = eval(read_file.read())
            if self.workshop_method == "hmw": 
                self.payload = params_dict['payload']["hmw"]
            elif self.workshop_method == "opposite": 
                self.payload = params_dict['payload']["opposite"]
            elif self.workshop_method == "bad idea": 
                self.payload = params_dict['payload']["bad"]
            elif self.workshop_method == "free text": 
                self.payload = params_dict['payload']["free"]
            
            # create a request from openai api
            self.raw_result = openai.Completion.create(**self.payload)

            # get idea list from raw result
            self.get_idea_list()
            return True
        else:
            """
            If api connection is not created, then return false.
            """
            return False
    
    def get_idea_list(self) -> list:
        """
        A function that get idea list from raw result.
        :return: A list of idea.
        """
        if self.raw_result is not None:
            for i in range(self.number_of_idea):
                self.idea_list.append(self.raw_result['choices'][i]['text'])
                self.idea_list_enhaced.append(self.raw_result['choices'][i]['text'])
        else:
            return False
        return self.idea_list
            
