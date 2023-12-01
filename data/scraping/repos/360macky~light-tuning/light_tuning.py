import openai
import requests
import os
from halo import Halo
from utils import remove_consecutive_newlines
from typing import Optional
import time
from prettytable import PrettyTable


class LightTuning:
    """
    The LightTuning class is responsible for generating a dataset, uploading it, and creating a fine-tuning job.
    """

    def __init__(self, api_key: str):
        """
        Initialize the LightTuning with the provided OpenAI API key.
        """
        self.api_key = api_key
        openai.api_key = self.api_key

    def generate_dataset(self, input_conversation: str, n_examples=64) -> Optional[str]:
        """
        Generate a dataset by creating a conversation using the GPT-4 model.
        The conversation is based on the provided input conversation.
        The generated dataset is written to a JSON file.
        """
        spinner = Halo(text='Generating dataset...', spinner='dots')
        spinner.start()
        try:
            dataset_messages = [
                {
                    "role": "system",
                    "content": f"""You are a generator of GPT-3.5 Turbo conversations that outputs JSONL formatted responses (where each JSON object is in only one line and all the JSONs objects separated by a new line) for a training dataset of a conversation between a user and an assistant.
                    
                    Your response should be in this format for every line!:
                    {{\\"messages\\" : [{{\\"role\\": \\"assistant\\", \\"content\\": \\"Instructions from assistant\\"}}, {{\\"role\\": \\"user\\", \\"content\\": \\"Question from user\\"}}, {{\\"role\\": \\"assistant\\", \\"content\\": \\"Reponse from assistant\\"}}
                    
                    YOU SHOULD NOT INCLUDE COMMAS AFTER EVERY LINE!
                    YOU SHOULD NOT INCLUDE THE BACKSLASHES IN YOUR RESPONSE, THEY ARE ONLY THERE TO ESCAPE THE QUOTES.
                    THE RESPONSE SHOULD BE IN JSONL FORMAT, WHERE EACH JSON OBJECT IS IN ONLY ONE LINE AND ALL THE JSONS OBJECTS SEPARATED BY A NEW LINE.
                    EVERY LINE SHOULD CONTAIN 1 SYSTEM, 1 USER AND 1 ASSISTANT MESSAGE.
                    
                    Your task is to follow the conversation as expected by the system prompt, including also the system, by many messages until reach {n_examples} lines of conversations.
                    
                    The conversation is this: {input_conversation}."""
                }
            ]
            completion = openai.ChatCompletion.create(
                model="gpt-4", messages=dataset_messages, temperature=0.7)

            # Convert the string to a JSON object
            # dataset = completion.choices[0].message['content']
            dataset = completion.choices[0].message['content']
            # Write the dataset to a JSON file

            # Use the function
            with open("dataset.json", "w") as f:
                dataset = remove_consecutive_newlines(dataset)
                f.write(dataset)
                lines_quantity = len(dataset.splitlines())
            spinner.succeed(
                f'Dataset generated successfully! You can access it at ./dataset.json and it has {lines_quantity} examples')
            return dataset
        except Exception as e:
            spinner.fail(f"Error in generate_dataset: {e}")
            return None

    def upload_dataset(self, file_path: str) -> Optional[str]:
        """
        Upload the dataset to OpenAI's servers.
        The dataset should be located at the provided file path.
        """
        spinner = Halo(text='Uploading dataset...', spinner='dots')
        spinner.start()
        try:
            with open(file_path, "rb") as f:
                headers = {
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
                }
                data = {
                    "purpose": "fine-tune"
                }
                files = {
                    "file": f
                }
                response = requests.post(
                    "https://api.openai.com/v1/files", headers=headers, data=data, files=files)
                response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
                spinner.succeed(
                    f'Dataset uploaded successfully! File ID: {response.json()["id"]}')
                return response.json()["id"]
        except Exception as e:
            spinner.fail(f"Error in upload_dataset: {e}")
            return None

    def create_fine_tuning_job(self, file_id: str) -> Optional[str]:
        """
        Create a fine-tuning job using the uploaded dataset.
        The dataset is identified by the provided file ID.
        """
        spinner = Halo(text='Creating fine-tuning job...', spinner='dots')
        # To wait until the file is ready
        time.sleep(8)
        spinner.start()
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            }
            data = {
                "training_file": file_id,
                "model": "gpt-3.5-turbo-0613"
            }
            response = requests.post(
                "https://api.openai.com/v1/fine_tuning/jobs", headers=headers, json=data)
            if response.status_code != 200:
                # Add color red to the text
                spinner.fail(f"The request to fine-tune the model failed!")
                print("Response: ", response.content)
                return None

            response.raise_for_status()
            spinner.succeed(
                'Fine-tuning job created successfully! The following is the response from OpenAI Fine-Tuning API:')

            # Create a PrettyTable object
            table = PrettyTable()

            # Add columns
            table.field_names = ["Property", "Value"]

            # Add rows
            response_json = response.json()
            for key, value in response_json.items():
                table.add_row([key, value])

            print(table)

            spinner.succeed(
                'You will receive an email when the fine-tuning job is finished!')

            return response.json()["id"]
        except Exception as e:
            spinner.fail(f"Error in create_fine_tuning_job: {e}")
            return None
