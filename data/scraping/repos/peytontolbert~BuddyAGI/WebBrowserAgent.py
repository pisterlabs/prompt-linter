import requests
import openai
import time
import llm.reason.prompt as ReasonPrompt
from memory.episodic_memory import EpisodicMemory, Episode
from memory.procedural_memory import ProceduralMemory
from memory.memory import MemoryManager
from gpt.chatgpt import ChatGPT
from typing import Dict, Any, Optional, Union, List
import os
import json
from dotenv import load_dotenv
import logging
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
class WebBrowserAgent:
    def __init__(self):
        self.gpt = ChatGPT()  # Initialize ChatGPT
        self.episodic_memory = EpisodicMemory()
        self.procedural_memory = ProceduralMemory()
        self.memory_manager = MemoryManager(self.gpt)
        self.messages = []
        self.tasks = {}
        #self.agents = [agent_class(self) for agent_class in agent_classes]
        pass
    def run(self):
        self.checkmessages()
        print(self.messages)
        while self.messages:
            message = self.messages.pop(0)  # Remove and return the first message
            self.handle_new_message(message)
        time.sleep(10)
    def checkmessages(self):
        try:
            response = requests.get("http://localhost:5000/browseragentmessages")
            if response.status_code == 200:
                self.messages.append(response.text)
            else:
                print("no new messages")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
    def sendreply(self, task_id, chat_id, user, task, action, complete_status=False):
        url = "http://localhost:5000/browseragentreply" if complete_status == False else "http://localhost:5000/completetask"
        data = {"task_id": task_id, "chat_id": chat_id, "user": user, "action": action, "task": task} if complete_status == False else {"task_id": task_id, "chat_id": chat_id, "user": user, "action": action, "task": task, "complete_status": True}
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                print("message sent")
            else:
                print("message not sent")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
    def parse_message(self, message):
        data = json.loads(message)
        task_id = data.get("task_id")
        chat_id = data.get("chat_id")
        user = data.get("user")
        task = data.get("message")
        return task, task_id, chat_id, user   
    def handle_new_message(self, user_message: Dict[str, Any]) -> None:
        task, task_id, chat_id, user = self.parse_message(user_message)
        interpreted_task = self.interpret_task(task)
        action = self.execute_task(interpreted_task, task_id)
        print(action)
        episode = Episode(
            message=user_message,
            result=interpreted_task,
            action=action

        )
        summary = self.episodic_memory.summarize_and_memorize_episode(episode)
        self.memory_manager.store_memory(user_message, interpreted_task, action, summary)
        self.save_agent()
        complete_status = self.verifytaskcompletion(interpreted_task, action)
        self.sendreply(task_id, chat_id, user, task=task, action=action, complete_status=complete_status )

    def interpret_task(self, task_description: str) -> str:

        # Incorporate episodic and tool information in the interpretation
        related_past_episodes = self.episodic_memory.remember_all_episode(
            task_description, k=3
        )
        tools = self.procedural_memory.remember_all_tools()
        tool_info = "\n".join([tool.get_tool_info() for tool in tools])

        prompt = ReasonPrompt.get_templatechatgpt(
            related_past_episodes=related_past_episodes,
            task=task_description,
            tool_info=tool_info
        )
        schematemplate = ReasonPrompt.add_schema_template()
        messages = [{"role": "system", "content": schematemplate}, {"role": "user", "content": prompt}]
        response = self.gpt.chat_with_gpt3(messages)
        interpreted_task = json.loads(response)
        return interpreted_task
    
    def generate_interpretation_prompt(self, task_description: str) -> str:
        # Generate a prompt for the GPT model to interpret the task
        template = """
        [TASK]
        {task_description}"""
        prompt = template.format(task_description=task_description)
        return prompt
    
    def execute_task(self, interpreted_task: Dict[str, Any]) -> None:
        # Determine the appropriate tool to execute the task
        tool_name, args = interpreted_task["action"]["tool_name"], interpreted_task["action"]["args"]
        
        action = self.act(tool_name, args)
        return action

    def verifytaskcompletion(self, task, action):
        systemprompt = """You are an autonomous task completion verification agent.
        Your role is to verify that the task has been completed successfully.
        Respond in JSON format following the example below:
        {
            'complete_status': True/False
        }
"""
        template = """[TASK]
        {task}
        [ACTION]
        {action}"""
        prompt = template.format(task=task, action=action)
        messages = [{"role": "system", "content": systemprompt}, {"role": "user", "content": prompt}]
        response = self.gpt.chat_with_gpt3(messages)
        parsed_message = json.loads(response)
        complete_status = parsed_message["complete_status"]
        return complete_status
    
    def act(self, tool_name: str, args: Dict) -> str:
        # Get the tool to use from the procedural memory
        try:
            tool = self.procedural_memory.remember_tool_by_name(tool_name)
        except Exception as e:
            return "Invalid command: " + str(e)
        try:
            print(f"args: " + str(args))
            result = tool.run(**args)
            return result
        except Exception as e:
            return "Could not run tool: " + str(e)
    def save_agent(self) -> None:
        episodic_memory_dir = f"{self.dir}/episodic_memory"
        filename = f"{self.dir}/agent_data.json"
        self.episodic_memory.save_local(path=episodic_memory_dir)

        data = {"name": self.agent_name,
                "episodic_memory": episodic_memory_dir
                }
        with open(filename, "w") as f:
            json.dump(data, f)
    def load_agent(self) -> None:
        absolute_path = self._get_absolute_path()
        if not "agent_data.json" in os.listdir(absolute_path):
            self.ui.notify("ERROR", "Agent data does not exist.", title_color="red")
        with open(os.path.join(absolute_path, "agent_data.json")) as f:
            agent_data = json.load(f)
            try:
                self.episodic_memory.load_local(agent_data["episodic_memory"])
            except Exception as e:
                self.ui.notify(
                    "ERROR", "Episodic memory data is corrupted.", title_color="RED")
                raise e
            else:
                self.ui.notify(
                    "INFO", "Episodic memory data is loaded.", title_color="GREEN")      
