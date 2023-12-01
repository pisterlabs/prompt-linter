from typing import List, Dict, Any
from datetime import datetime
from src.logger_utils import create_logger
from src import constants

from langchain import BasePromptTemplate, LLMChain
from langchain.schema import BaseMemory
from langchain.base_language import BaseLanguageModel
from pydantic import BaseModel
from src.data.session_manager import SessionManager

from src.llms.prompts.memory_templates import SUMMARIZER_PROMPT, ENTITY_EXTRACTION_PROMPT, \
    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT, SENTIMENT_ANALYSIS_PROMPT

import asyncio

MESSAGE_BUFFER_LENGTH = 10
SENTIMENT_HISTORY_LENGTH = 3
ENTITY_HISTORY_LENGTH = 3
KNOWLEDGE_HISTORY_LENGTH = 3
SUMMARY_HISTORY_LENGTH = 3

class DynamicMemory(BaseMemory, BaseModel):
    logger = create_logger(__name__, constants.SYSTEM_LOG_FILE)
    """Memory class for storing information."""
    summary_llm: BaseLanguageModel
    entity_llm: BaseLanguageModel
    knowledge_llm: BaseLanguageModel
    sentiment_llm: BaseLanguageModel
    summarizer_prompt: BasePromptTemplate = SUMMARIZER_PROMPT
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
    knowledge_extraction_prompt: BasePromptTemplate = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
    sentiment_analysis_prompt: BasePromptTemplate = SENTIMENT_ANALYSIS_PROMPT

    session: SessionManager = None
    ai_name: str = "Govinda"

    session_messages: List = []
    message_buffer: List = []
    chat_history_string: str = ""
    current_knowledge: str = ""
    current_entities: str = ""
    current_sentiment: str = ""

    current_input: str = ""
    current_output: str = ""

    current_summary: str = ""
    previous_summary: str = ""

    memory_string: str = ""
    memory_key: str = "dynamic_memory"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("Dynamic Memory: Initializing")
        self.session = kwargs.get('session', None)
        self.ai_name = kwargs.get('ai_name', "Govinda")
        self.setup()
        self.logger.info("Dynamic Memory: Initialized")
    
    def setup(self):
        self.logger.info("Dynamic Memory: Setting Up Dynamic Memory")
        pass

    def clear(self):
        self.memory_string = ""
        self.current_summary = ""
        self.previous_summary = ""
        self.current_knowledge = ""
        self.current_entities = ""
        self.current_sentiment = ""
        self.current_input = ""
        self.current_output = ""
        self.session_messages = []
        self.message_buffer = []
        self.chat_history_string = ""

    @property
    def memory_variables(self) -> List[str]:
        """Define the variables we are providing to the prompt."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return self.memory_string

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        self.memory_string = self.generate_new_memory()

    def preprocessing(self, input):
        self.logger.info("Dynamic Memory: Preprocessing")
        self.current_input = input
        loop = asyncio.get_event_loop()
        try:
            self.current_sentiment, self.current_entities = loop.run_until_complete(asyncio.gather(
                self.generate_new_sentiment_analysis(),
                self.generate_new_entities()
            ))
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")

    def postprocessing(self, input):
        self.logger.info("Dynamic Memory: Postprocessing")
        self.current_input = input
        loop = asyncio.get_event_loop()
        try:
            self.current_summary, self.current_knowledge, = loop.run_until_complete(asyncio.gather(
                self.generate_new_summary(),
                self.generate_new_knowledge()
            ))
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")

    async def generate_new_summary(self) -> str:
        self.logger.info("Generating new summary")
        new_summary = "Default summary."
        summarizer_chain = LLMChain(llm=self.summary_llm, prompt=self.summarizer_prompt)
        if len(self.message_buffer) > SUMMARY_HISTORY_LENGTH:
            new_summary = await summarizer_chain.arun(user_name = self.session.current_user.display_name, 
                                                      ai_name=self.ai_name, 
                                                      current_summary=self.current_summary, 
                                                      chat_history=self.get_last_k_messages(SUMMARY_HISTORY_LENGTH))
        self.previous_summary = self.current_summary
        self.current_summary = new_summary
        self.logger.info("Summary generation complete")
        return new_summary

    def generate_new_chat_history(self) -> str:
        self.logger.info("Generating new chat history")
        new_chat_history = ""
        for message in self.message_buffer:
            speaker = message.get("speaker", "")
            content = message.get("content", "")
            if speaker and content:
                new_chat_history += f"{speaker.capitalize()}: {content}\n"
        self.chat_history_string = new_chat_history
        return new_chat_history

    async def generate_new_entities(self) -> str:
        self.logger.info("Generating new entities")
        new_entities = "None"
        chain = LLMChain(llm=self.entity_llm, prompt=self.entity_extraction_prompt)
        if len(self.message_buffer) > ENTITY_HISTORY_LENGTH:
            new_entities = await chain.arun(user_name=self.session.current_user.display_name, 
                                            ai_name=self.ai_name, 
                                            history=self.get_last_k_messages(ENTITY_HISTORY_LENGTH), 
                                            input=self.current_input)
        self.logger.info("New entity generation complete")
        return new_entities

    async def generate_new_sentiment_analysis(self) -> str:
        self.logger.info("Generating new sentiment analysis")
        new_sentiment_analysis = "None"
        chain = LLMChain(llm=self.sentiment_llm, prompt=self.sentiment_analysis_prompt)
        if len(self.message_buffer) > SENTIMENT_HISTORY_LENGTH:
            new_sentiment_analysis = await chain.arun(user_name=self.session.current_user.display_name, 
                                                      ai_name=self.ai_name, 
                                                      history=self.get_last_k_messages(SENTIMENT_HISTORY_LENGTH), 
                                                      input=self.current_input)
        self.logger.info("New sentiment analysis generation complete")
        return new_sentiment_analysis

    async def generate_new_knowledge(self) -> str:
        self.logger.info("Generating new knowledge triplets")
        new_knowledge = "None"
        chain = LLMChain(llm=self.knowledge_llm, prompt=self.knowledge_extraction_prompt)
        if len(self.message_buffer) > KNOWLEDGE_HISTORY_LENGTH:
            new_knowledge = await chain.arun(user_name=self.session.current_user.display_name, 
                                             ai_name=self.ai_name, 
                                             history=self.get_last_k_messages(KNOWLEDGE_HISTORY_LENGTH), 
                                             input=self.current_input)
        self.logger.info("New knowledge triplets generation complete")
        return new_knowledge

    def add_message(self, role: str, content: str, speaker: str, time: datetime) -> None:
        self.logger.info("Adding message to buffer")
        message = {"role": role, "content": content, "speaker": speaker, "time": time}
        if len(self.message_buffer) > MESSAGE_BUFFER_LENGTH:
            if self.message_buffer[0]["role"] == "system" and self.message_buffer[1]["role"] == "system":
                self.message_buffer.pop(0)
            if self.message_buffer[0]["role"] != "system":
                self.message_buffer.pop(0)
            else:
                self.message_buffer.pop(1)
        self.session_messages.append(message)
        self.message_buffer.append(message)
        self.logger.info("Message added to buffer")

    def get_last_message(self) -> str:
        if len(self.session_messages) > 0:
            last_message = self.message_buffer[-1]
            return last_message["speaker"] + ": " + last_message["content"]
        else:
            return ""

    def get_last_k_messages(self, k: int) -> str:
        if len(self.session_messages) > 0:
            last_k_messages = self.session_messages[-k:]
            return "\n".join([message["speaker"] + ": " + message["content"] for message in last_k_messages])
        else:
            return ""



