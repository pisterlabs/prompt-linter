'''This script shows a strategy implementation for the openai RAG assistant
The "retrieval" tool used for RAG is only free to use for a limited time
After November 2023 the tool will be paid for as price/gb/month for storage
And also price/assistant
Consult the openai pricing page for more information
'''

import time
from typing import List
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from dataclasses import dataclass

from strategy import BaseConfig, IStrategy, SessionManager, StrategyRegistry

load_dotenv()
# openai.api_key = os.getenv("OPENAI_KEY")
openai_api_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=openai_api_key)

@dataclass
class RAGConfig(BaseConfig):
    assistant_id: str
    thread_id: str
    model: str = "gpt-4-1106-preview"  # or any other model you prefer


class RAGRetrievalStrategy(IStrategy):

    def execute(self, query: str):
        ##TODO: documents and context must be uploaded before assistant and threads are created!

        ## check if assistant exists, else create
        if not self.config.assistant_id:
            assistant = self.create_assistant()
            self.assistant = assistant
            self.config.assistant_id = assistant.id

        # Are we executing against an existing thread?
        if not self.config.thread_id:
            thread = client.beta.threads.create()
            self.config.thread_id = thread.id
        else:
            thread = client.beta.threads.retrieve(
                thread_id=self.config.thread_id
            )

        # Add the user's message to the thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query,
        )

        # Create a run to get a response from the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.config.assistant_id,
        )

        # Implement logic to wait for the run to complete and fetch the response
        while run.status != "completed":
            time.sleep(2)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )

        response = client.beta.threads.messages.list(thread_id=thread.id)

        return response.data[0].content[0].text.value
    
    def create_assistant(self):
        '''assistant must be aware of file ids so it must be created after files have been uploaded'''
        assistant = client.beta.assistants.create(
            name="RAGAssistant",
            instructions="Answer questions using provided context.",
            model=self.config.model,
            tools=[{"type": "retrieval"}],
            file_ids=self.config.file_ids,
        )
        self.assistant = assistant
        self.config.assistant_id = assistant.id
        return assistant

    def upload_file(self, file_path: str):    
        file = client.files.create(
            file=open(file_path, "rb"),
            purpose="assistants",
        )

        if hasattr(self.config, "file_ids"):
            self.config.file_ids.append(file.id)
        else:
            self.config.file_ids = [file.id]

        return file.id
    
    def upload_files(self, file_paths: List[str]):
        file_ids = []
        for file_path in file_paths:
            file_id = self.upload_file(file_path)

        return file_ids
    
    def delete_files(self, file_ids: List[str] = None):
        if file_ids is None:
            file_ids = self.config.file_ids

        for file_id in file_ids:
            client.files.delete(file_id=file_id)

    def delete_thread(self, thread_id: str = None):
        if thread_id is None:
            thread_id = self.config.thread_id

        client.beta.threads.delete(thread_id=thread_id)

    def delete_assistant(self, assistant_id: str = None):
        if assistant_id is None:
            assistant_id = self.config.assistant_id

        client.beta.assistants.delete(assistant_id=assistant_id)

    def cleanup(self):
        self.delete_files()
        self.delete_thread()
        self.delete_assistant()

        
    
if __name__ == "__main__":

    registry = StrategyRegistry()
    ## use a config with no assistant_id or thread_id (they will be created)
    default_config = RAGConfig(name="RAGRetrievalStrategy", assistant_id=None,thread_id=None)
    registry.register("RAGRetrievalStrategy", RAGRetrievalStrategy, default_config)

    session_manager = SessionManager(registry)

    # Example usage:
    session = session_manager.create_session("RAGRetrievalStrategy", session_id="unique_session_id")
    session.strategy.upload_files(["test_input.txt"])
    response = session.execute_strategy("What car does Abraham Seldom drive?")
    print(response)


    ## pattern loading a previous session and carrying on
    session_manager.save_session(session)
    session_loaded = session_manager.load_session("unique_session_id")  
    response = session.execute_strategy("What else can you tell me about Abraham Seldom?")
    print(response)

    ## pattern requesting a session that already exists
    session_manager.save_state(file_name="session_manager.pkl")
    new_session_manager = SessionManager(registry)
    new_session_manager.load_state(file_name="session_manager.pkl")
    session_existing = new_session_manager.get_or_create_session("RAGRetrievalStrategy", "unique_session_id")
    response = session.execute_strategy("Tell me a story about Abraham Seldom?")
    print(response)

    # cleanup all openai resources
    session.strategy.cleanup()