import requests
import openai
import time
import llm.reason.prompt as ReasonPrompt
from memory.episodic_memory import EpisodicMemory, Episode
from memory.procedural_memory import ProceduralMemory
from memory.semantic_memory import SemanticMemory
from memory.memory import MemoryManager
from gpt.chatgpt import ChatGPT
from typing import Dict, Any, Optional, Union, List
import psycopg2
import os
import json
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
class DBAgent:

    def __init__(self):
        self.gpt = ChatGPT()  # Initialize ChatGPT
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()
        self.memory_manager = MemoryManager(self.gpt)
        self.messages = []
        pass

    def run(self):
        self.checkmessages()
        print(self.messages)
        while self.messages:
            result, message = self.filtermessages()
            print(result)
            if result == 'yes' or 'Yes':
                print("DB agent handling database message")
                self.handlemessages(message)
            else:
                print("message not meant for database")
        time.sleep(10)


    def checkmessages(self):
        try:
            response = requests.get("http://localhost:5000/dbagentmessages")
            if response.status_code == 200:
                self.messages.append(response.text)
            else:
                print("no new messages")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
    
    def filtermessages(self, retries=5, delay=10):
        if not self.messages:
            print("No messages to filter.")
            return
        #get the oldest message from the self.messages list
        message = self.messages.pop(0)
        systemprompt = """Remember, your only reply should be yes or no."""
        inputprompt = """You are an AI database filter agent.
        Your task is to determine if the message is meant for tasks handling a database. 
        Reply yes or no if the follow message is meant to interact with a database:
        [MESSAGE]
        {message}"""
        chat_input = inputprompt.format(message=message)
        
        for i in range(retries):
            try:
                results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": chat_input}])
                result =  str(results['choices'][0]['message']['content'])
                return result, message
            except openai.error.ServiceUnavailableError:
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    raise

    def handlemessages(self, message):
        related_past_episodes = self.episodic_memory.remember_related_episodes(
            message, k=3
        )
        tools = self.procedural_memory.remember_all_tools()
        tool_info = ""
        for tool in tools:
            tool_info += tool.get_tool_info() + "\n"
        Dicts = {"related_past_episodes": related_past_episodes, "task": message, "tool_info": tool_info}
        prompt = ReasonPrompt.get_templatechatgpt(
            Dicts=Dicts
        )
        schematemplate = ReasonPrompt.add_schema_template()
        prompt_messages = [{"role": "system", "content": schematemplate}, {"role": "user", "content": prompt}]
        results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=prompt_messages)
        result = results['choices'][0]['message']['content']
        print(result)
        print("parsing message")
        parsed_message = json.loads(result)

        tool_name = parsed_message["action"]["tool_name"]
        print(f"tool_name: {tool_name}")
        args = parsed_message["action"]["args"]
        #get tool from result
        action = self.act(tool_name, args)
        print(action)
        episode = Episode(
            message=message,
            result=result,
            action=action

        )
        summary = self.episodic_memory.summarize_and_memorize_episode(episode)
        self.memory_manager.store_memory(message, result, action, summary)
        self.save_agent()
        return action

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


    def connect_to_database():
        try:
            # Connect to your PostgreSQL database. Replace these placeholders with your actual database credentials
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("USER"),
                password=os.getenv("PASSWORD"),
                host=os.getenv("HOST"),
                port=os.getenv("PORT")
            )
            return conn
        except Exception as e:
            print(f"An error occurred while connecting to the database: {e}")
            return None
                


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




    def superprompt(self, message):
        systemprompt = """You are an AI designated to reword a prompt to be used for a Database Agent Tool to interact with a database.
        Your task is to take a given message and to reply with an efficient database control. Follow the example:
        [EXAMPLE]
        User: Please create a database called 'DBAgent'
        AI: CREATE DATABASE DBAgent;"""
