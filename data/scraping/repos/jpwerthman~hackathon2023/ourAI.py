import openai
import json
import os
import dotenv
import pandas as pd
from pandas import json_normalize
dotenv.load_dotenv()

key = os.getenv("OPENAI_API_KEY")

openai.api_key = key

class Bot:
    def __init__(self):
        self.transaction_data = ''' ''' # The rest of your transaction data here

    def get_transaction_info(self):
        return self.transaction_data
    
    def set_transaction_info(self, data):
        print("New transaction data set")
        self.transaction_data = data

    def _format_system_message(self, user_message):
        template = f'''
              [Financial Manager Mode Activated]
                Objective: Assist the user in interpreting their transactional data based on the provided CSV data. Respond with relevant transaction details.
            {user_message}

            Data:
            {self.transaction_data} 

            You will respond like you are talking to them. For example: if they ask, "tell me about my transfers", you will respond with details about the transfers.
            '''
        return template

    def run_conversation(self, initial_message):
        system_message = self._format_system_message(initial_message)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": initial_message}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )

        response_content = response.choices[0].message["content"]

        if "get_transaction_info" in response_content:
            messages.extend([
                {"role": "assistant", "content": response_content},
                {
                    "role": "function",
                    "name": "get_transaction_info",
                    "content": self.get_transaction_info()
                }
            ])
            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
            )
            return second_response.choices[0].message["content"]

        return response_content
    
    def JSONtoCSV(self, data):
        # Normalize the json data
        df = json_normalize(data)

        # Selected columns
        selected_columns = ['amount', 'description', 'postedDate', 
                            'transactionDate', 'createdDate', 
                            'categorization.normalizedPayeeName', 
                            'categorization.category', 
                            'categorization.bestRepresentation', 
                            'categorization.country']
        new_df = df[selected_columns]
        csv_string = new_df.to_csv(index=False)
        print("Length", len(csv_string))

        # Truncate CSV data if too long
        max_length = 2048  # Adjust this value based on your needs
        if len(csv_string) > max_length:
            return csv_string[-max_length:]
        return csv_string