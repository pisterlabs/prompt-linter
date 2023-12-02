import openai
from stringcolor import *
from dotenv import load_dotenv
import os
import tiktoken

# loads the .env file that contains the API key
load_dotenv() 

# hidden api key
key = os.getenv("KEY")

#
class ChatGPTChat:
    def __init__(
        self,
        api_key=key, # The API key for OpenAI
        model="gpt-3.5-turbo", #  The ChatGPT model used (gpt-3.5-turbo is an example, can be replaced)
        max_tokens=1000, # Maximum tokens allowed for response length
        temperature=0.5, # 0.0 is deterministic, 1.0 is creative
        completions=1, # Number of completions per prompt
        presence_penalty=0.5, # The higher the value, the more likely new topics will be introduced
        frequency_penalty=0.5,  # The higher the value, the more likely information will be repeated
    ):
        self.api_key = api_key
        self.model = model  # The ChatGPT model used (gpt-3.5-turbo is an example, can be replaced)
        self.max_tokens = max_tokens  # Maximum tokens allowed for response length
        self.temperature = temperature  # 0.0 is deterministic, 1.0 is creative
        self.completions = completions  # Number of completions per prompt
        self.presence_penalty = presence_penalty  # The higher the value, the more likely new topics will be introduced
        self.frequency_penalty = frequency_penalty  # The higher the value, the more likely information will be repeated
        openai.api_key = self.api_key   # Pass API key to OpenAI

    def get_chatgpt_response(self, messages):
        response = openai.ChatCompletion.create(
            model=self.model,  # ChatGPT model
            messages=messages,  # Ongoing conversation
            temperature=self.temperature,  # Chosen temperature (creativity/strictness)
            max_tokens=self.max_tokens,  # Maximum tokens allowed
            n=self.completions,  # Number of completions
            presence_penalty=self.presence_penalty,  # Chosen new-topics-probability
            frequency_penalty=self.frequency_penalty,  # Chosen repeat-info-probability
        )
        choices = [
            choice.message["content"] for choice in response.choices
        ]  # Extracting the content of responses
        return choices

    # Define function to count tokens
    def count_tokens(self, text):
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = len(enc.encode(text))
        token_usage_costs = tokens / 1000 * 0.015 # 0.015 is the price per token, calculates the costs for the prompt
        return token_usage_costs

    def chat_interface(self, user_input):
        # Display welcome message
        print(cs("ChatGPT is working on your responses!", "blue")) 
        #print(cs("Type 'quit' to exit the chat.\n", "darkblue"))
        system_role = "You are a helpful expert for jobsearch and application. You adapt the wording in regards to the job the user wants to apply for and provide additional information on companies and competitive market salaries."  # DEFINE SYSTEM ROLE HERE
        global messages
        messages = [{"role": "system", "content": system_role}]
        messages.append({"role": "user", "content": user_input})

        # Append user's input to messages
        responses = self.get_chatgpt_response(messages)
        prompt_costs = self.count_tokens(str(messages))  # Count tokens
        return prompt_costs, responses


if __name__ == "__main__":
    chat_gpt = ChatGPTChat()
    chat_gpt.chat_interface(
        [] # enter prompts as string to test the gpt from this file
    )
