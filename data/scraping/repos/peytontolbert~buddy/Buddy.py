import os
import json
import time
import traceback
from utils import Stack # Assuming you have a package for ChatGPT, Emotion, and DBManager
from memory.episodic_memory import EpisodicMemory, Episode
from memory.procedural_memory import ProceduralMemory
from memory.semantic_memory import SemanticMemory
from action.action import ActionManager
from tasks.tasks import TaskManager
from goals.goals import GoalManager
from state.state import StateManager
import llm.reason.prompt as ReasonPrompt
from llm.reason.schema import JsonSchema as ReasonSchema
from emotion.emotion import Emotion
from thoughts.thoughts import ThoughtManager
from filemanager.filemanager import FileManager
from db.db import DBManager
from gpt.chatgpt import ChatGPT
from ui.base import BaseHumanUserInterface
from ui.cui import CommandlineUserInterface
from langchain.llms.base import BaseLLM
from langchain import LLMChain
from llm.json_output_parser import LLMJsonOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union, List
from gpt.chatgpt import ChatGPT
import openai
from memory.memory import MemoryManager
import requests
from collections import deque

# Define the default values
DEFAULT_AGENT_NAME = "FinnAGI"
DEFAULT_AGENT_DIR = "./agent_data"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define the schema for the llm output
REASON_JSON_SCHEMA_STR = json.dumps(ReasonSchema.schema)

class FinnAGI(BaseModel):
    """
    This is the main class for the Agent. It is responsible for managing the tools and the agent.
    """
    # Define the tools
    dir: str = Field(
        DEFAULT_AGENT_DIR, description="The folder path to the directory where the agent data is stored and saved")
    agent_name: str = Field(DEFAULT_AGENT_NAME, description="The name of the agent")
    ui: BaseHumanUserInterface = Field(
        CommandlineUserInterface(), description="The user interface for the agent")
    gpt: Optional[ChatGPT] # add this line to declare gpt as an optional field
    db: Optional[DBManager]  # Initialize DB Manager
    thought_stack: Optional[Stack]  # Initialize thought stack
    procedural_memory: ProceduralMemory = Field(
        ProceduralMemory(), description="The procedural memory about tools agent uses")
    episodic_memory: EpisodicMemory = Field(
        None, description="The short term memory of the agent")
    semantic_memory: SemanticMemory = Field(
        None, description="The long term memory of the agent")
    file_manager: Optional[FileManager]
    task_manager: TaskManager = Field(
        None, description="The task manager for the agent")
    action_manager: Optional[ActionManager]
    thought_manager: Optional[ThoughtManager]
    memory_manager: Optional[MemoryManager]
    state_manager: Optional[StateManager]
    working_memory: Optional[deque]
    thoughts: Optional[List[Any]]
    messages: Optional[Any]
    chat: Optional[Any]
    last_task: Optional[Any]
    class Config:
        arbitrary_types_allowed = True
    #name: str = Field(DEFAULT_AGENT_NAME, description="The name of the agent")
    #role: str = Field(DEFAULT_AGENT_ROLE, description="The role of the agent")
    #goal: str = Field(DEFAULT_AGENT_GOAL, description="The goal of the agent")
    #llm: BaseLLM = Field(..., description="llm class for the agent")
    #openaichat: Optional[ChatOpenAI] = Field(
    #    None, description="ChatOpenAI class for the agent")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.episodic_memory = EpisodicMemory() # Initialize episodic memory
        self.semantic_memory = SemanticMemory() # Initialize semantic memory
        self.file_manager = FileManager(self)  # Create an instance of FileManager
        self.task_manager = TaskManager()
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
        self.gpt = ChatGPT()  # Initialize ChatGPT
        self.db = DBManager()  # Initialize DB Manager
        self.thought_stack = Stack()  # Initialize thought stack
        self.action_manager = ActionManager()
        self.memory_manager = MemoryManager(self.gpt)
        self.working_memory = deque(maxlen=12)
        self.thoughts = []
        self.messages = []
        self.last_task = []
        self.procedural_memory = ProceduralMemory() # Initialize procedural memory
        self.thought_manager = ThoughtManager(self)  # Create an instance of ThoughtManager  
        self.state_manager = StateManager()  # Create an instance of StateManager   
        self.chat = {}
        self.agent_name = "BuddyAGI"
    def run(self):
        self.check_messages()
        if len(self.messages) > 0:
            message = self.messages[-2:].copy()
            self.messages.clear()
            qa = self.task_manager.clarify(message)
            print(qa)
            full_request = self.send_message(qa)
            with self.ui.loading("Generate Task Plan..."):
                self.task_manager.generate_task_plan(
                    message=full_request 
                )
            self.ui.notify(title="ALL TASKS",
                            message=self.task_manager.get_incomplete_tasks_string(),
                            title_color="BLUE")                
            while True:
                incomplete_tasks = self.task_manager.get_incomplete_tasks()
                current_task = self.task_manager.get_current_task_string()
                if len(incomplete_tasks) > 0:
                #ReAct: Reasoning
                    with self.ui.loading("Thinking..."):
                        try:
                            reasoning_result = self._reason()
                            if reasoning_result is not None:
                                thoughts = reasoning_result["thoughts"]
                                action = reasoning_result["action"]
                                tool_name = action["tool_name"]
                                args = action["args"]
                            else:
                                print("No reasoning result found")
                        except Exception as e:
                            raise Exception("An error occurred: " + str(e) + "\n" + traceback.format_exc())
                    self.ui.notify(title="TASK", message=thoughts.get("task", "No task found"))
                    self.ui.notify(title="IDEA", message=thoughts.get("idea", "No idea found"))
                    self.ui.notify(title="REASONING", message=thoughts.get("reasoning", "No reasoning found"))
                    self.ui.notify(title="CRITICISM", message=thoughts.get("criticism", "No criticism found"))
                    self.ui.notify(title="THOUGHT", message=thoughts.get("summary", "No summary found"))
                    self.ui.notify(title="NEXT ACTION", message=action)
                    # Task Complete
                    if tool_name == "task_complete":
                        action_result = args["result"]
                        self._task_complete(action_result)
                        # save agent data
                        with self.ui.loading("Save agent data..."):
                            self.save_agent()
                        self.working_memory.append(str(action_result))
                    # Action with tools
                    else:
                        try:
                            action_result = self._act(tool_name, args)
                        except Exception as e:
                            raise e
                        self.ui.notify(title="ACTION RESULT", message=action_result)
                        action_result_string = str(action_result)
                    episode = Episode(
                        thoughts=thoughts,
                        action=action,
                        result=action_result_string,
                        summary=action_result_string
                    )
                    # Store memory
                    self.memory_manager.store_memory(self.working_memory, thoughts, action, episode.summary)
                    summary = self.episodic_memory.summarize_and_memorize_episode(episode)
                    self.ui.notify(title="MEMORIZE NEW EPISODE",
                                message=summary, title_color="blue")
                    episode_str = str(episode)
                    entities = self.semantic_memory.extract_entity(episode_str)
                    self.ui.notify(title="MEMORIZE NEW KNOWLEDGE",
                                message=entities, title_color="blue")
                    self.working_memory.append(action_result_string)
                    self.task_manager.eval_action(action_result_string, current_task)
                    time.sleep(1)
                    self.save_agent()
                            #self.volition(self.agent_name, self.agent_goal)
                        # Perform the selected action
                        #self.action_manager.perform_action(action, thought)
                else:
                    self.task_manager.save_completed_tasks()
    def simulate_sleep(self):
        thoughts = self.thought_stack
    def _reason(self) -> Union[str, Dict[Any, Any]]:
        current_task_description = self.task_manager.get_current_task_string()
        if current_task_description is None:
            return None
        else:  
            # Retrie task related memories
            with self.ui.loading("Retrieve memory..."):
                # Retrieve memories related to the task.
                related_past_episodes = self.episodic_memory.remember_related_episodes(
                    current_task_description,
                    k=3)
                if related_past_episodes is not None and len(related_past_episodes) > 0:
                    try:
                        self.ui.notify(title="TASK RELATED EPISODE",
                                    message=related_past_episodes)
                    except Exception as e:
                        print(e)
                # Retrieve concepts related to the task.
                if current_task_description is None:
                    return None
                else:
                    if len(current_task_description) > 0:
                        related_knowledge = self.semantic_memory.remember_related_knowledge(
                            current_task_description,
                            k=3
                        )
                        if related_knowledge is None:
                            related_knowledge = "No related knowledge."
                                                    # Get the relevant tools
                            # If agent has to much tools, use "remember_relevant_tools"
                            # because too many tool information will cause context windows overflow.
                            tools = self.procedural_memory.remember_all_tools()

                            # Set up the prompt
                            tool_info = ""
                            for tool in tools:
                                tool_info += tool.get_tool_info() + "\n"

                            # Get the recent episodes
                            memory = self.episodic_memory.remember_recent_episodes(2)

                            # If OpenAI Chat is available, it is used for higher accuracy results.
                            if current_task_description is not None and len(current_task_description) > 0:
                                propmt = ReasonPrompt.get_chat_template(
                                    memory=memory, 
                                    related_past_episodes=related_past_episodes,
                                    related_knowledge=related_knowledge,
                                    task=current_task_description,
                                    tool_info=tool_info
                                )
                                prompt = str(propmt)
                                results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": prompt}])
                                result =  str(results['choices'][0]['message']['content'])

                            else:
                               return None
                            # Parse and validate the result
                            try:
                                result_json_obj = LLMJsonOutputParser.parse_and_validate(
                                    json_str=result,
                                    json_schema=REASON_JSON_SCHEMA_STR,
                                    gpt=self.gpt
                                )
                                return result_json_obj
                            except Exception as e:
                                raise Exception(f"Error: {e}")

                        else:
                            if len(related_knowledge) > 0:
                                self.ui.notify(title="TASK RELATED KNOWLEDGE",
                                    message=related_knowledge)
                            # Get the relevant tools
                            # If agent has to much tools, use "remember_relevant_tools"
                            # because too many tool information will cause context windows overflow.
                            tools = self.procedural_memory.remember_relevant_tools(current_task_description)

                            # Set up the prompt
                            tool_info = ""
                            for tool in tools:
                                tool_info += tool.get_tool_info() + "\n"

                            # Get the recent episodes
                            memory = self.episodic_memory.remember_recent_episodes(2)
                            Dicts = {"related_past_episodes":related_past_episodes,"related_knowledge":related_knowledge,"task":current_task_description,"tool_info":tool_info}
                            # If OpenAI Chat is available, it is used for higher accuracy results.
                            propmt = ReasonPrompt.get_templatechatgpt(
                                memory=memory, 
                                Dicts=Dicts
                            )
                            prompt = str(propmt)
                            memoryprompt = ReasonPrompt.memory_to_template(
                                memory=memory,
                            )
                            if memoryprompt is not None:
                                prompt += memoryprompt
                            schematemplate = ReasonPrompt.add_schema_template()
                            results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": schematemplate},{"role": "user", "content": prompt}])
                            result = results['choices'][0]['message']['content']
                            try:
                                result_json_obj = LLMJsonOutputParser.parse_and_validate(
                                    json_str=result,
                                    json_schema=REASON_JSON_SCHEMA_STR
                                )
                                return result_json_obj
                            except Exception as e:
                                raise Exception(f"Error: {e}")
    def _act(self, tool_name: str, args: Dict) -> str:
        # Get the tool to use from the procedural memory
        try:
            tool = self.procedural_memory.remember_tool_by_name(tool_name)
        except Exception as e:
            return "Invalid command: " + str(e)
        try:
            result = tool.run(**args)
            return result
        except Exception as e:
            return "Could not run tool: " + str(e)
    def modify_task(self, task):
        new_task = self.task_manager.modify_current_task(task)
        return new_task
    def _task_complete(self, result: str) -> str:
        current_task = self.task_manager.get_current_task_string()
        self.last_task = current_task
        self.ui.notify(title="COMPLETE TASK",
                       message=f"TASK:{current_task}\nRESULT:{result}",
                       title_color="BLUE")

        self.task_manager.complete_current_task(result)
        return result
    def format_output_data(self, data):
        # Your logic for formatting output data here.
        formatted_data = data
        return formatted_data
    def process_input(self, thought, emotional_state, goal):
        system_prompt = "Processing user input"
        processed_input = self.gpt.process_thought(system_prompt, thought, emotional_state, goal)
        self.thoughts.append(processed_input)
        return processed_input
    def save_state(self):
        self.state_manager.save_agent_state(self)  # Save agent state
    def load_state(self):
        return self.state_manager.load_agent_state()  # Load agent state
    def update_memory(self, thought):
    # Your logic to update memory with the new thought. This could involve storing the thought in a database or a local variable.
    # For example, if you want to append the thought to the working_memory deque:
        self.working_memory.append(thought)
    def check_messages(self):
        messages = ""
        response = requests.get("http://localhost:5000/buddymessages")
        if response.status_code == 200:
            self.messages.append(response.text)
        else:
            print("no new messages")
        return
    def send_message(self, message):
        messages = [message]
        messagestring = str(message)
        print(f"sending message", message)
        headers = {'Content-type': 'application/json'}
        response = requests.post("http://localhost:5000/messagetocreator", json={"user": "Buddy", "message":messagestring}, headers=headers)
        print(f"POST request status code: {response.status_code}")
        if response.status_code == 200:
            while True:
                response = requests.get("http://localhost:5000/buddymessages")
                print(f"GET request status code: {response.status_code}")
                if response.status_code == 200:
                    response_data = response.json()
                    print(f"Response data: {response_data}")
                    messages.append(response_data['message'])
                    return messages
                elif response.status_code == 404:
                    print("no new messages")
                    time.sleep(2)
                else:
                    print(f"Unexpected status code: {response.status_code}")
                    break
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
            self.agent_name = agent_data["name"]
            self.agent_goal = agent_data["goal"]

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
