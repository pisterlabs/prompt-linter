#!/usr/bin/python3  # Shebang line for specifying Python 3 interpreter

import os  # Import the os module for accessing environment variables
import openai  # Import the OpenAI library for AI functionalities

# Importing classes from other modules in the project
from imageCapture import Image_manipulation
from compare_ranges import compare_ranges
from json_to_dict import load_json

class Extract():
    # Constructor of the class Extract
    def __init__(self):
        # Retrieving and setting the OpenAI API key from environment variables
        self.api_key = os.environ["OPENAI_API_KEY"]
        openai.api_key = self.api_key

        # Initializing variables to store data
        self.numbers = []  # List to store numbers
        self.numbers_buffer = []  # Temporary buffer to store numbers
        self.count = 0  # Counter to keep track of numbers

        # List of keys (blood test parameters)
        self.keys = ['WHITE BLOOD CELL', 'RED BLOOD CELL', 'HEMOGLOBIN', 'HEMATOCRIT',
                     'MCV', 'MCH', 'MCHC', 'RDW', 'RDW-SD', 'PLATELET COUNT (PLT)', 'MPV',
                     'NEUTROPHILS', 'LYMPHOCYTES', 'MONOCYTES', 'EOSINOPHILS', 'BASOPHILS',
                     'NRBC', 'IG', 'NEUTROPHILS ABSOLUTE VALUE',
                     'LYMPHOCYTES ABSOLUTE VALUE', 'MONOCYTES ABSOLUTE VALUE',
                     'EOSINOPHILS ABSOLUTE VALUE', 'BASOPHILS ABSOLUTE VALUE', 'NRBC',
                     'IG']
        self.json_path = 'bloodiqjson.json'  # Path to JSON file with reference data

    def process_data(self):
        # Method to process raw data and extract numbers
        res = Image_manipulation.dataProcessing(Image_manipulation.rawData)
        for item in res:
            # Ignore 'HIGH' or 'LOW' values
            if item in ['HIGH', 'LOW']:
                continue
            # Check if the item is a number and process it
            elif item.replace('.', '', 1).isdigit():
                self.count += 1
                self.numbers_buffer.append(float(item))
                # Every third number, reset the buffer and counter
                if self.count == 3:
                    self.numbers.append(self.numbers_buffer[0])
                    self.numbers_buffer = []
                    self.count = 0
            else:
                # If buffer is not empty, append the first number
                if self.numbers_buffer:
                    self.numbers.append(self.numbers_buffer[0])
                self.numbers_buffer = []
                self.count = 0
        return self.numbers

    def create_dict(self):
        # Method to create a dictionary from the processed data
        results = dict(zip(self.keys, self.process_data()))
        return results

    def compare(self):
        # Method to compare the processed data against reference ranges
        ranges = load_json(self.json_path)
        comparison = compare_ranges(self.create_dict(), ranges)
        return comparison

    def generate_prompt(self):
        # Method to generate a prompt for OpenAI's model based on the comparison results
        combined_values = ' '.join(f"{key}: {value}" for key, value in self.compare().items())
        script = "Having these blood test values in High and Low might mean: "
        combined_prompt = script + "{answer by ChatGPT} And our advice would be: What advice can you give me based on these parameters? \n" + combined_values
        return combined_prompt

    def generate_response(self):
        # Method to generate a response from OpenAI's model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Specify the model
            messages=[
                {"role": "system", "content": self.generate_prompt()}
            ],
            max_tokens=1000,  # Limit the response length
            temperature=0.5  # Control the randomness of the response
        )
        generated_response = response['choices'][0]['message']['content']
        return generated_response

# Main block to execute if the script is run directly
if __name__ == "__main__":
    extract = Extract()  # Instantiate the Extract class
    print("Generated Response:", extract.generate_response())  # Print the response from OpenAI's model
