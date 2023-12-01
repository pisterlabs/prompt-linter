import re
import json
from langchain.chat_models import ChatOpenAI
from langchain.llms import Anthropic
from langchain.schema import HumanMessage, SystemMessage
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY #Import API Keys stored in a separate file. You can do this with envionrment variables as well.
import datetime
from pathlib import Path


# At the moment langchain API wrappers are needed due to the separation of chat models and language models. These wrappers allow us to use the same interface for both.
# Class to communicate with OpenAI for generating responses. Wrapped around the langchain wrappers
class OpenAIModel():
    def __init__(self, openai_api_key, **model_params):
        """Initialize the OpenAI chat model.

        Parameters:
        openai_api_key (str): API key to access OpenAI API
        model_params (dict): Parameters to configure the model like temperature, n, etc.
        """
        self.chat = ChatOpenAI(openai_api_key=openai_api_key, **model_params, request_timeout=120)
    
    def __call__(self, request_messages):
        return self.chat(request_messages).content
    
    def bulk_generate(self, message_list):
        return self.chat.generate(message_list)

# Class to communicate with claude-v1.3 for generating responses. Wrapped around the langchain wrappers
class AnthropicModel():
    def __init__(self, anthropic_api_key, **model_params):
        """Initialize the Anthropic chat model.

        Parameters:
        anthropic_api_key (str): API key to access Anthropic API
        model_params (dict): Parameters to configure the model like model_name, max_tokens, etc.
        """
        self.chat = Anthropic(model=model_params['model_name'],temperature=model_params['temperature'], max_tokens_to_sample=model_params['max_tokens'], anthropic_api_key=anthropic_api_key)
    
    def __call__(self, request_messages):
        # Convert request_messages into a single string to be used as preamble
        # This is a hacky solution to the fact that the langchain wrapper expects a single string as input. 
        # But the performance is actaualy really good especially with the XML formatting method.
        message = "\n\n".join([message.content for message in request_messages])
        return self.chat(message)
    
    def bulk_generate(self, message_list):
        new_message_list = []
        for request_messages in message_list:
            new_message = "\n".join([message.content for message in request_messages])
            new_message_list.append(new_message)
        return self.chat.generate(new_message_list)


class LanguageExpert: 
    """Defines an AI assistant/expert for natural language generation.  

    Attributes:
    name (str): Name of the expert
    system_message (str): Expert's initial greeting message 
    description (str): Description of the expert's abilities
    example_input (str): Sample user input the expert can handle 
    example_output (str): Expert's response to the sample input
    model_params (dict): Parameters to configure the language model
    """
    def __init__(self, name: str, system_message=None, description=None,  
                 example_input=None, example_output=None, model_params=None):  

        ## Initialize expert attributes##
        self.name = name  
        self.system_message = system_message
        self.description = description 
        self.example_input = example_input 
        self.example_output = example_output  
        
        ##Set default model parameters if none provided##
        if model_params is None:  
            model_params = {"model_name": "claude-v1.3", "temperature":  0.00,  
                            "frequency_penalty": 1.0, "presence_penalty":  0.5,  
                            "n": 1, "max_tokens":  512}
        self.model_params = model_params
        self.gen_chat()  #Generate the chat object to get model-specific responses

    def serialize(self): 
        """Returns a JSON-serializable representation of the expert.

        Returns: 
        dict: Contains all expert attributes.
        """
        return {
            "name": self.name,
            "system_message": self.system_message,
            "description": self.description,
            "example_input": self.example_input,
            "example_output": self.example_output,
            "model_params": self.model_params
        }

    def get_content(self):  
        """Returns the expert definition in an fake XML format.

        Returns:
        SystemMessage: Expert definition wrapped in XML tags.  
        """
        example_output = self.example_output
        example_input = self.example_input

        content = '<assistant_definition>\n'

        if self.name:
            content += f'<name>{self.name}</name>\n'

        if self.description:
            content += f'<role>{self.description}</role>\n'

        if self.system_message:
            content += f'<system_message>{self.system_message}</system_message>\n'

        if example_input:
            content += f'<example_input>{example_input}</example_input>\n'

        if example_output:
            content += f'<example_output>{example_output}</example_output>\n'

        content += '</assistant_definition>'

        content = SystemMessage(content=content)
        return content
    
    def generate(self, message): 
        """Generates a response to the input message. 

        Passes the input through the chat model and returns its response.
        
        Parameters:
        message (str): User's input message
        
        Returns: 
        response (str): expert's response to the message
        """ 
        human_message = HumanMessage(content=message)
        request_message = [self.get_content(), human_message]
        response  = self.chat(request_message)
        self.log([message], [response])
        return response

    def log(self, requests, responses):
        """Logs a conversation between the user and the expert.

        Parameters:
        requests (list): List of user requests/messages
        responses (list): List of expert responses 
        """
        now = datetime.datetime.now()
        filename = Path(f'./logs/{now.strftime("%Y-%m-%d_%H-%M-%S")}_{self.name}.txt')
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        log = f'Expert Name: {self.name}\n\nRequests:\n'
        for request in requests: 
            log += f'{request}\n\n'
        log += 'Responses:\n'
        for response in responses:
            log += f'{response}\n\n'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(log)
    
    def extract_texts_from_generations(self, generations):
        """Extracts plain text responses from a list of generated responses.

        Parameters: 
        generations (list): List of generated responses from the model

        Returns:
        list: List of plain text responses
        """   
        return [generation[0].text for generation in generations]

    def bulk_generate(self, messages:list):
        """Generates responses for multiple input messages.

        Parameters: 
        messages (list): List of user input messages

        Returns: 
        responses (list): List of corresponding expert responses
        """
        human_messages = [HumanMessage(content=message) for message in messages]
        request_messages = [[self.get_content(), human_message] for human_message in human_messages]
        responses = self.chat.bulk_generate(request_messages)
        responses = self.extract_texts_from_generations(responses.generations)
        self.log(messages, responses)
        return responses
    
    def __call__(self, message:str): 
        """Allows the expert to be called like a function.

        Invokes the generate() method.
        """
        return self.generate(message)

    def change_param(self, parameter_name, new_value):
        """Changes a expert definition parameter to a new value.

        Updates the internal model_params dictionary and regenerates 
        the chat object.

        Parameters:
        parameter_name (str): Name of the parameter to change
        new_value: New value for the parameter 
        """
        if parameter_name in ["model_name", "temperature", "frequency_penalty", "presence_penalty", "n", "max_tokens"]:
            self.__dict__["model_params"][parameter_name] = new_value
        else:
            self.__dict__[parameter_name] = new_value
        self.gen_chat()
    
    def gen_chat(self): 
        """Instantiates the chat object used to generate responses.

        The chat object is either an AnthropicModel or OpenAIModel, depending 
        on the model_name parameter. 
        """
        if 'gpt' in self.model_params["model_name"]:
            self.chat = OpenAIModel(openai_api_key=OPENAI_API_KEY, **self.model_params)
        elif 'claude' in self.model_params["model_name"]:
            self.chat = AnthropicModel(anthropic_api_key=ANTHROPIC_API_KEY, **self.model_params)
        else:
            raise 'Model not supported'
