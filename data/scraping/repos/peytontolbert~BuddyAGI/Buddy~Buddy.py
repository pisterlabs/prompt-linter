import requests
import openai
import time
import llm.reason.prompt as ReasonPrompt
from memory.episodic_memory import EpisodicMemory, Episode
from memory.semantic_memory import SemanticMemory
from memory.custom_memory import CustomMemory
from filemanager.filemanager import FileManager
from ui.base import BaseHumanUserInterface
from ui.cui import CommandlineUserInterface
from typing import Dict, Any, List
from pydantic import Field
import os
import json
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DEFAULT_AGENT_DIR = "./agent_data"
class Buddy():
    ui: BaseHumanUserInterface = Field(
        CommandlineUserInterface(), description="The user interface for the agent")

    def __init__(self):
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.custom_memory = CustomMemory()
        self.messages = []
        self.tasks = []
        self.activetasks = []
        self.completedtasks = []
        self.ui=CommandlineUserInterface()
        self.dir = DEFAULT_AGENT_DIR
        self.agent_name = "Buddy"
        self.file_manager = FileManager(self)  # Create an instance of FileManager
        self.file_manager._get_absolute_path()  # Get absolute path
        self.file_manager._create_dir_if_not_exists()
        if self.file_manager._agent_data_exists():
            load_data = self.ui.get_binary_user_input(
                "Agent data already exists. Do you want to load the data?\n"
                "If you choose 'Yes', the data will be loaded.\n"
                "If you choose 'No', the data will be overwritten."
            )
            if load_data:
                self.file_manager.load_agent()
            else:
                self.ui.notify("INFO", "Agent data will be overwritten.")
        pass

    def run(self):
        self.checkmessages()
        print(self.messages)
        while self.messages:
            print(f"self.messages: {self.messages}")
            message = self.messages.pop(0)  # Remove and return the first message
            user = message['user']
            chat = message['messages']
            chat_id = message['chat_id']
            print(f"user: {user}")
            print(f"chat_id: {chat_id}")
            result, message = self.filtermessages(chat)
            print(result)
            if result == 'CONVERSATIONAL':
                print("conversational agent handling message")
                context = self.findcontext(message)
                reply = self.handleconversation(message, context)
                self.sendreply(reply, user, chat_id)
                self.save_agent()
            elif result == 'TASK-ORIENTED':
                print("task agent handling message")
                newtask = self.generatetask(chat, user, chat_id)
                reply = f"Task created with task_id: {newtask['task_id']}, Message to agents: {newtask['chat']}"
                self.save_agent()
        self.messages.clear()
        time.sleep(10)

    def findcontext(self, message):
        entities = self.semantic_memory.remember_related_knowledge(message, k=3)
        related_past_episodes = self.episodic_memory.remember_related_episodes(
            message, k=3
        ) 
        knowledge_base = self.custom_memory.remember_related_episodes(
            message, k=3
        )
        schematemplate = """You are an autonomous context agent.
        Your task is to provide any context based on past episodes, entities, and a custom knowledge base. If no context is found, respond with 'No context found'."""
        prompt = """
        [ENTITIES]
        {entities}
        [RELATED PAST EPISODES]
        {related_past_episodes}
        [KNOWLEDGE BASE]
        {knowledge_base}
        [MESSAGE]
        {message}"""
        chat_input = prompt.format(entities=entities, related_past_episodes=related_past_episodes, knowledge_base=knowledge_base, message=message)
        prompt_messages = [{"role": "system", "content": schematemplate}, {"role": "user", "content": chat_input}]
        results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=prompt_messages)
        result = results['choices'][0]['message']['content']
        print(f"context: {result}")
        return result

    def checkmessages(self, complete_status = False):
        try:
            response = requests.get("http://localhost:5000/helpermessages")
            if response.status_code == 200:
                data = response.json()
                print(f"data: {data}")
                chats = data['messages']
                for chat_object in chats:
                    for chat_id, chat_data in chat_object.items():
                        print(f"new mesages: {chat_object}")
                        user = chat_data['user']
                        messages = chat_data['messages']
                        message = {"user": user, "messages": messages, "chat_id": chat_id }
                        self.messages.append(message)
            else:
                print("no new messages")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
    
    def filtermessages(self, chat, retries=5, delay=10):
        if not chat:
            print("No messages to filter.")
            return
        #get the oldest message from the self.messages list
        message = chat
        inputprompt = """You are a helpful autonomous intent filter.
        Your task is to determine the message's intent between three choices: CONVERSATIONAL/TASK-ORIENTED. 
        Only respond with the correct intent [CONVERSATIONAL/TASK-ORIENTED] following message:
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

    def handleconversation(self, message, context=None):
        systemprompt = """You are an autonomous conversational agent.
        Your task is to respond to a message in a conversational manner.
        You will be provided context to help you respond.
        Follow the example:
        [EXAMPLE]
        User: Hello, how are you?
        AI: I am fine, how are you?
        User: Do you remember what the project name of that article I sent you yesterday was?
        AI: Yes, it was called 'Project Buddy'."""
        inputprompt = """You are an autonomous conversational agent.
        Your task is to respond to a message in a conversational manner.
        [CONTEXT]
        {context}
        Follow the example:
        [MESSAGE]
        {message}"""
        chat_input = inputprompt.format(message=message, context=context)
        results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": systemprompt}, {"role": "user", "content": chat_input}])
        result = results['choices'][0]['message']['content']
        print(f"AI: {result}")
        return result

    def generatetask(self, chat, user, chat_id):
        message = f"{user}: {chat}"
        related_past_episodes = self.episodic_memory.remember_related_episodes(
            message, k=3
        )
        related_knowledge = self.semantic_memory.remember_related_knowledge(
            message, k=3
        )
        Dicts = {"related_past_episodes": related_past_episodes, "related_knowledge": related_knowledge, "task": message}
        prompt = ReasonPrompt.get_templatechatgpt2(
            Dicts=Dicts
        )
        schematemplate = ReasonPrompt.add_schema_template2()
        prompt_messages = [{"role": "system", "content": schematemplate}, {"role": "user", "content": prompt}]
        results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=prompt_messages)
        result = results['choices'][0]['message']['content']
        print(result)
        print("parsing message")
        parsed_message = json.loads(result)
        messagetoagents = parsed_message["action"]["task"]
        task_id = self.sendnewtask(messagetoagents, chat_id, user)
        self.activetasks[task_id] = {
            "task": messagetoagents,
            "chat_id": chat_id,
            "user": user,
            "messages": [{"role": "user", "content": f"User: {messagetoagents}"}]
        }
        episode = Episode(
            message=message,
            result=result,
            action=messagetoagents
        )
        self.episodic_memory.summarize_and_memorize_episode(episode)
        return self.activetasks[task_id]
    
    def sendnewtask(self, message, chat_id, user):
        # Base URL for the requests
        base_url = 'http://localhost:5000'
        # Data to be sent in the POST request
        data = {'user': 'Buddy', 'message': message, 'chat_id': chat_id, 'sender': user }
        response = requests.post(f'{base_url}/managernewtask ', json=data)
        # Optional: Check response status and handle accordingly
        if response.status_code == 200:
            data = response.json()
            return data['task_id']
        else:
            print(f"Failed to send new task")
            return f"Failed to send new task"

    def sendreply(self, reply, user, identifier):
        url = "http://localhost:5000/buddychat"
        data = {"message": reply, "receiver": user, "chat_id": identifier}
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                print("message sent")
            else:
                print("message not sent")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def save_agent(self) -> None:
        episodic_memory_dir = f"{self.dir}/episodic_memory"
        semantic_memory_dir = f"{self.dir}/semantic_memory"
        filename = f"{self.dir}/agent_data.json"
        self.episodic_memory.save_local(path=episodic_memory_dir)
        self.semantic_memory.save_local(path=semantic_memory_dir)

        data = {"name": self.agent_name,
                "episodic_memory": episodic_memory_dir,
                "semantic_memory": semantic_memory_dir
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
                self.semantic_memory.load_local(agent_data["semantic_memory"])
            except Exception as e:
                self.ui.notify(
                    "ERROR", "Semantic memory data is corrupted.", title_color="red")
                raise e
            else:
                self.ui.notify(
                    "INFO", "Semantic memory data is loaded.", title_color="GREEN")
            try:
                self.episodic_memory.load_local(agent_data["episodic_memory"])
            except Exception as e:
                self.ui.notify(
                    "ERROR", "Episodic memory data is corrupted.", title_color="RED")
                raise e
            else:
                self.ui.notify(
                    "INFO", "Episodic memory data is loaded.", title_color="GREEN")


