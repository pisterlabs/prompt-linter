import json
import random
import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def choose_model_source():
    while True:
        source = input("Do you want to use a 'local' or 'online' model? ").lower()
        if source in ['local', 'online']:
            return source
        else:
            print("Invalid input. Please enter 'local' or 'online'.")

# Ask the user to choose the model source
model_source = choose_model_source()

if model_source == 'local':
    openai.api_base = os.getenv('LOCAL_API_BASE')
    openai.api_key = os.getenv('LOCAL_API_KEY')
    model = os.getenv('LOCAL_MODEL_PATH')
else:
    # Assign OpenAI API key from environment variable
    openai.api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('OPENAI_ENGINE')


class Agent:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.memory = self.load_memory()
        self.ideology = self.generate_ideology()
        self.temperature = 0.8

    def load_memory(self):
        filename = self.name.replace(" ", "_") + '.json'
        try:
            with open(filename, 'r') as fp:
                data = json.load(fp)
            self.name = data.get("name", self.name)
            self.ideology = data.get("ideology", self.generate_ideology())
            return data.get("memory", {})
        except FileNotFoundError:
            return {}



    def save_memory(self):
        filename = self.name.replace(" ", "_") + '.json'
        data = {
            "name": self.name,
            "ideology": self.ideology,
            "memory": self.memory
        }
        with open(filename, 'w') as fp:
            json.dump(data, fp)


    def call_ai_api(self, prompt, max_tokens=1000, temperature=1, presence_penalty=1, frequency_penalty=1.0, n=1, echo=False, stream=False):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            n=n,
            echo=echo,
            stream=stream
        )
        return response.choices[0].text.strip()

    def generate_ideology(self):
        system_instruction = f"Acting strictly in character as {self.name} describe how your defining attributes and"
        prompt_agent = f"{system_instruction} your unique perspective guide your decisions."
        return self.call_ai_api(prompt_agent, max_tokens=200, temperature=1, presence_penalty=1.5, frequency_penalty=1.2)

    def respond_to_prompt(self, user_prompt, shared_memory, summary=None):
        sentiment_of_question = self.analyze_sentiment(user_prompt)
        other_responses = [f"{name}: {conv['agent_response']}" for name, conv in shared_memory.items() if name != self.name]
        other_responses_context = ". ".join(other_responses)

        # Retrieve previous discussions from memory
        past_discussions = ". ".join([f"Question: {q}. Response: {a['agent_response']}" for q, a in list(self.memory.items())[-1:]])  # Retrieves last 1 discussions

        system_instruction = f"Considering your '{self.ideology}', your attitude towards the subject '{sentiment_of_question}', and your past discussions '{past_discussions}', answer the following question from your unique perspective as {self.name}. Do not create lists or respond with a question."
        prompt_user = f"{system_instruction} Consider the question: '{user_prompt}'"
        response_user = self.call_ai_api(prompt_user, max_tokens=1000, temperature=self.temperature, presence_penalty=1, frequency_penalty=0.8)
        sentiment_of_response = self.analyze_sentiment(response_user)
        self.adjust_temperature(sentiment_of_response)
        return response_user


    def analyze_sentiment(self, text):
        system_instruction = f"Analyze the sentiment of the following response by {self.name}: '{text}'."
        response_sentiment = self.call_ai_api(system_instruction, max_tokens=100, temperature=0.7, presence_penalty=1, frequency_penalty=0.8)
        if 'positive' in response_sentiment.lower():
            return 'positive'
        elif 'negative' in response_sentiment.lower():
            return 'negative'
        else:
            return 'neutral'

    def adjust_temperature(self, sentiment):
        if sentiment == 'positive':
            self.temperature = max(0.9, self.temperature + 0.1)
        elif sentiment == 'negative':
            self.temperature = min(0.9, self.temperature - 0.2)
        else:
            self.temperature = 0.6

def conduct_round_table_discussion():
    # Set up agents
    number_of_agents = int(input("Enter the number of agents you want in the discussion: "))
    
    # Dictionary of possible agent names
    agent_names_dict = ["a skeleton", "a witch", "a wizard", "a stoic", "the left handed path", "a pragmatist", "a ghost", "a vampire", "a werewolf", "a zombie"]
    
    agents = []
    for i in range(number_of_agents):
        agent_name = input(f"Enter the name of agent {i+1} (leave blank for random selection): ")
        
        # If the user doesn't enter a name, select a random one from the dictionary
        if not agent_name:
            agent_name = random.choice(agent_names_dict)
        
        agents.append(Agent(agent_name, model))

    # Conduct discussion
    while True:
        user_prompt = input("Enter a discussion prompt or 'quit' to exit: ")
        if user_prompt.lower() == 'quit':
            break

        shared_memory = {}

        for agent in agents:
            print(f"\nAgent: {agent.name}\n")
            agent_response = agent.respond_to_prompt(user_prompt, shared_memory)
            print(f"\nResponse:\n {agent_response}\n")
            shared_memory[agent.name] = {"agent_response": agent_response}
        
        # Update and save agent memories
        for agent in agents:
            agent.memory[user_prompt] = shared_memory[agent.name]
            agent.save_memory()


if __name__ == "__main__":
    conduct_round_table_discussion()
