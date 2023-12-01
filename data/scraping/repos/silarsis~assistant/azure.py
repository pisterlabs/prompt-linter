from typing import Callable
from langchain.llms import AzureOpenAI
import openai


class ModelClass(abc.ABC):
    def __init__(self, name: str, prompt_template: str):
        
        self.llm = AzureOpenAI(
            deployment_name="gpt-35-turbo", model_name="gpt-35-turbo")
    
    def update_prompt_template(self, new_template: str):
        # Update the prompt template
        pass
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None]):
        # Write a prompt to the bot and callback with the response.
        callback(self.prompt(prompt))
    
    def prompt(self, prompt: str):
        # Write a prompt to the bot and return the response.
        # return self.llm(prompt)

        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo", # replace this value with the deployment name you chose when you deployed the associated model.
            messages = [{
                "role": "system",
                "content": "You are an AI classifier to classify logistic package weights for a deliveries driver to carry. Small means the driver can carry one set with a plastic bag and one hand easily. Medium means the driver can only carry max 1 set with his two hands. Bulky means the driver needs a trolley or other tools. The input format is json, where item field if the item description and quantity is the quantity of the described item. Only put one of ['\''small'\'', '\''medium'\'', '\''bulky'\''] in your answer."
            },
            {
                "role": "user",
                "content": "[{'\''item'\'': '\''300 ml shampoo'\'', '\''quantity'\'': 1}, {'\''item'\'': '\''tooth brush'\'', '\''quantity'\'': 2}, {'\''item'\'': '\''towel'\'', '\''quantity'\'': 2}]"
            }],
            temperature=0,
            max_tokens=350,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        return response