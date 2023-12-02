'''
Copyright 2023 Starburst Data

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import openai

import env

from dataModels import Data

import pandas as pd
pd.options.plotting.backend = "plotly"

import env

from tabulate import tabulate

class OpenAI():
    '''Class to handle OpenAI API calls
    Attributes
    ----------
        api_key : str
            OpenAI API key
        model : str
            OpenAI model name
        data_class : dataModels.Data
            Data class for the model
        system_message : str
            System message for OpenAI chatbot
        
    Methods
    -------
        get_models()
            Get list of OpenAI models
        save_settings(api_key, model)
            Save settings for the session.
        set_system_message()
            Set system message for OpenAI chatbot.
        predict(message, system_message = None)
            Predict response from OpenAI chatbot from the supplied question'''

    def __init__(self, data_class: Data, model = None, api_key = None) -> None:
        '''Initialize OpenAI class with data class and model name
        Args:  data_class : dataModels.Data class
                model: OpenAI model name = None
                api_key: OpenAI API key = None
        '''
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = env.OPENAI_API_KEY
        
        openai.api_key = self.api_key
        
        if model is None:
            self.model = env.OPENAI_MODEL
        else:
            self.model = model
    
        self.data_class = data_class

        self.models = self.get_models()

    def get_models(self):
        '''Get list of OpenAI models'''
        models = openai.Model.list(self.api_key)
        print(models)
        return models

    def save_settings(self, api_key: str, model: str):
        '''Save settings for the session.
        Args:  api_key: OpenAI API key
                model: OpenAI model name'''
        self.api_key = api_key
        self.model = model

    def set_system_message(self):
        '''Set system message for OpenAI chatbot.'''
        
        #message_data_type = 'csv'
        #message_data = self.data_class.df_summary.to_pandas().rename(columns={'state': 'State', 'risk_appetite': 'Risk_Appetite', 'count': 'Count_of_Customers'}).to_csv(index=False)
        
        message_data_type = 'table'
        message_data = tabulate(self.data_class.df_summary.to_pandas().rename(columns={'state': 'State', 'risk_appetite': 'Risk_Appetite', 'count': 'Count_of_Customers'}), headers='keys', tablefmt='outline', showindex=False)
        
        self.system_message = f'''You are an AI assistant who's purpose is to provide information on structured data.
                                The data formated as {message_data_type} is: 
                                {message_data}"'''

    def predict(self, message, system_message = None):
        '''Predict response from OpenAI chatbot from the supplied question
        Args:  message: Question to ask the OpenAI chatbot
                system_message = None: System message for OpenAI chatbot'''
        
        if system_message is None:
            system_message = self.set_system_message()
        
        if system_message and self.system_message is None:
            raise Exception("System message is not defined. Please use set_system_message() method to set system message.")
        
        if env.DEBUG: print(self.system_message)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                "role": "system",
                "content": self.system_message
                },
                {
                "role": "user",
                "content": message
                }
            ],
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n = 3
        )
        responses = [i.message.content + "\n\n" for i in response.choices]
        if env.DEBUG: print(responses)
        return responses[0]
    