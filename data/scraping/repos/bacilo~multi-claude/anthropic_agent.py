import anthropic

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

API_KEY = config.get('api', 'key')

class Agent:
    def __init__(self, api_key, model="claude-2"): # claude-v1.3
        self.api_key = api_key
        self.model = model
        self.client = anthropic.Client(api_key)
        self.conversation_history = ""

    def get_ai_response(self, prompt):
        max_tokens_to_sample = 150

        full_prompt = f"{self.conversation_history}{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"

        response = self.client.completion(
            prompt=full_prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=self.model,
            max_tokens_to_sample=max_tokens_to_sample,
        )

        return response['completion'].strip(), response['truncated']

    def get_last_ai_response(self):
        last_ai_response = self.conversation_history.rsplit(anthropic.AI_PROMPT, 1)[-1].rsplit(anthropic.HUMAN_PROMPT, 1)[0].strip()
        return last_ai_response

    def update_conversation_history(self, human_input, ai_output):
        self.conversation_history += f"{anthropic.HUMAN_PROMPT} {human_input} {anthropic.AI_PROMPT} {ai_output} {anthropic.HUMAN_PROMPT}"

class AgentManager:
    def __init__(self):
        self.agents = {}
        self.current_agent = "default"
        self.agents[self.current_agent] = Agent(API_KEY)

    def create_agent(self, name_and_prompt=None):
        if name_and_prompt is None:
            return "Please provide an agent name."

        name_and_prompt_parts = name_and_prompt.split(None, 1)
        name = name_and_prompt_parts[0]
        initial_prompt = name_and_prompt_parts[1] if len(name_and_prompt_parts) > 1 else None

        if name in self.agents:
            return f"Agent '{name}' already exists."

        self.agents[name] = Agent(API_KEY)
        self.current_agent = name
        message = f"Created and switched to agent '{name}'."

        if initial_prompt:
            ai_response, truncated = self.agents[name].get_ai_response(initial_prompt)
            self.agents[name].update_conversation_history(initial_prompt, ai_response)
            message += f"\n\nInitial Prompt: {initial_prompt}\nAI Response: {ai_response}"

        return message



    def switch_agent(self, name):
        if name in self.agents:
            self.current_agent = name
            return f"Switched to agent '{name}'."
        return f"Agent '{name}' does not exist."

    def agents_conversation(agent_manager, input_string):
        input_parts = input_string.split(maxsplit=2)
        if len(input_parts) != 3:
            return "Invalid input. Please provide two agent names and an initial prompt."

        agent1, agent2, initial_prompt = input_parts
        if agent1 not in agent_manager.agents or agent2 not in agent_manager.agents:
            return "Both agent names must be valid."

        response1, _ = agent_manager.agents[agent1].get_ai_response(initial_prompt)
        agent_manager.agents[agent1].update_conversation_history(initial_prompt, response1)
        print(f"{agent1} says: {response1}")

        num_turns = int(input("Enter the number of turns: "))

        for _ in range(num_turns - 1):
            prompt1 = agent_manager.agents[agent1].conversation_history[-1][1]  # Get last AI response for agent1
            response1, _ = agent_manager.agents[agent2].get_ai_response(prompt1)
            agent_manager.agents[agent2].update_conversation_history(prompt1, response1)
            print(f"{agent2} says: {response1}")

            prompt2 = agent_manager.agents[agent2].conversation_history[-1][1]  # Get last AI response for agent2
            response2, _ = agent_manager.agents[agent1].get_ai_response(prompt2)
            agent_manager.agents[agent1].update_conversation_history(prompt2, response2)
            print(f"{agent1} says: {response2}")

    def list_agents(agent_manager, _=None):
        agent_list = ', '.join(agent_manager.agents.keys())
        if agent_list:
            return f"Current agents: {agent_list}"
        else:
            return "There are no agents created yet."

    def remove_agent(self, name):
        if name in self.agents:
            del self.agents[name]
            if self.current_agent == name:
                self.current_agent = "default"
            return f"Deleted agent '{name}'."
        return f"Agent '{name}' does not exist."

    def get_current_agent(self):
        return self.agents[self.current_agent]
