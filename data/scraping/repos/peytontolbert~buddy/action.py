import openai
import os
from dotenv import load_dotenv
import llm.reason.prompt as ReasonPrompt
import random
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
class ActionManager:
    choose_action_prompt = """
    I am an impassioned autonomous artifical intelligence. 
    
    Here is my current thought: 
    "{thought}"

    Here is my current task:
    {task}

    Answer only in first person:
    """

    select_action_prompt = """
    I am an impassioned autonomous artificial intelligence.

    Here is my current goal and emotional state:
    {task}
    {thought}
    {best_next_action}

    Among the following actions, which ones seem to be the most appropriate for me to take?

    Possible actions:
    {
        "Action-11": "Search the internet for information", 
        "Action-22": "Message Creator",
        "Action-33": "Think creatively",
        "Action-44": "Recall something or remember context",
        "Action-55": "Chill out & wait",
        "Action-66": "the best action is not listed"
    }

    Only answer in a schema format. For example, if you think Action-11 and Action-22 are the most appropriate actions, then your answer should be:
    {
        "Action-11": "Search the internet for information",
        "Action-22": "Message Creator"
    }
    
    """
    def __init__(self):
        self.current_action = None
        self.possible_actions =     {
        "Action-11": "Search the internet for information", 
        "Action-22": "Message Creator",
        "Action-33": "Think creatively",
        "Action-44": "Recall something or remember context",
        "Action-55": "Chill out & wait",
        "Action-66": "the best action is not listed"
        }     # This is a placeholder. Replace with your actual implementation.

    
    def prompt_for_action(self, thought, task):

        
        # If OpenAI Chat is available, it is used for higher accuracy results.
        if self.openaichat:
            propmt = ReasonPrompt.get_chat_template(memory=memory).format_prompt(
                name=self.name,
                role=self.role,
                goal=self.goal,
                related_past_episodes=related_past_episodes,
                related_knowledge=related_knowledge,
                task=current_task_description,
                tool_info=tool_info
            ).to_messages()
            result = self.openaichat(propmt).content



        chat_input = self.choose_action_prompt.format(thought=thought, task=task)
        chat_output = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": chat_input}])
        return chat_output.choices[0].message['content']

    def select_action(self, thought, task):
        best_next_action = self.prompt_for_action(thought, task)
        suitable_actions = self.get_possible_actions(task, thought, best_next_action)
        print(suitable_actions)
        return suitable_actions


    def is_suitable(self, action):
        # Add your logic here to evaluate the suitability of an action
        pass

    def get_possible_actions(self, task, thought, best_next_action):
        chat_input = self.select_action_prompt.format(task=task, thought=thought, best_next_action=best_next_action)
        chat_output = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": chat_input}])
        return chat_output.choices[0].message['content']


    def perform_action(self, action, thought):
        if action == "Action-11":
            # Perform action 11: Search the internet for information
            print("Performing action 11: Search the internet for information")
        elif action == "Action-22":
            print("Performing action 22: Ask a question to my creator")
            prompt = """
            I am an impassioned autonomous artificial intelligence.
            I have chosen to ask a question to my creator.
            Here is my thought process:
            {thought}
            Only respond with question(s) for the creator.
            """
            chat_input = prompt.format(thought=thought)
            chat_output = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": chat_input}])
            print(chat_output.choices[0].message['content'])
        elif action == "Action-33":
            # Perform action 33: Think creatively
            print("Performing action 33: Think creatively")
        elif action == "Action-44":
            # Perform action 44: Send a message to [undefined]
            print("Performing action 44: Send a message to [undefined]")
        elif action == "Action-55":
            # Perform action 55: Recall something or remember context
            print("Performing action 55: Recall something or remember context")
        elif action == "Action-66":
            # Perform action 66: Chill out & wait
            print("Performing action 66: Chill out & wait")
        elif action == "Action-77":
            # Perform action 77: the best action is not listed
            print("Performing action 77: the best action is not listed")
        else:
            print("Unknown action")
        return print("Action performed: ", action)
    
    def reflect_on_action(self, action, thought):
            # Add your logic here to reflect on the action
        prompt = """I am an impassioned autonomous artificial intelligence.
        I have taken action {action}.
        my thoughts during the action were:
        {thought}
        I am now reflecting on the action:"""
        chat_input = prompt.format(action=action, thought=thought)
        chat_output = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": chat_input}])
        print(chat_output.choices[0].message['content'])
        return chat_output.choices[0].message['content']

