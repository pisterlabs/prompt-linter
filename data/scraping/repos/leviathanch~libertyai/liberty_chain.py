import json
import re
import hashlib

from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Sequence,
    List,
    Union,
)

from pydantic import (
    BaseModel,
    Extra,
    Field,
    root_validator
)

from datetime import datetime
from sentence_transformers import util
from langdetect import detect

import langchain

from langchain.chains.llm import LLMChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.memory import PostgresChatMessageHistory

from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseMemory
from langchain.agents.agent import AgentExecutor
from langchain.vectorstores.pgvector import PGVector

from LibertyAI.liberty_prompt import EN_PROMPT, DE_PROMPT
from LibertyAI.liberty_llm import LibertyLLM
from LibertyAI.liberty_embedding import LibertyEmbeddings

class LibertyChain(LLMChain, BaseModel):

    human_prefix: str = "Human"
    ai_prefix: str = "LibertyAI"
    hash_table: dict = {}
    prompt: BasePromptTemplate = EN_PROMPT
    mrkl: AgentExecutor = None
    memory: BaseMemory = None
    summary: ConversationSummaryMemory = None
    user_name: str = ""
    user_mail: str = ""
    embeddings: LibertyEmbeddings = None
    vectordb: PGVector = None
    pghistory: PostgresChatMessageHistory = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prep outputs."""
        self._validate_outputs(outputs)
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}

    def start_generations(self, message):
        encoded1 = self.embeddings.embed_query(message)
        encoded2 = self.embeddings.embed_query("What's the weather in X?")
        context = ""

        if self.mrkl:
            if util.pytorch_cos_sim(encoded1, encoded2)[0] > 0.5:
                context = self.mrkl.run(message)
        #else:
        #    documents = self.vectordb.similarity_search_with_score(query=message, k=1)
        #    context = documents[0][0].page_content

        if self.memory:
            chat_history = self.memory.load_memory_variables(inputs=[])['history']
        else:
            chat_history = ""

        if self.summary:
            chat_summary = self.summary.load_memory_variables(inputs=[])['history']
        else:
            chat_summary = ""

        d = {
            'input': message,
            'history': chat_history,
            'context': context,
            'summary': chat_summary,
            'current_date': datetime.now().strftime("%A (%d/%m/%Y)"),
            'current_time': datetime.now().strftime("%H:%M %p"),
            #'user_name': self.user_name,
            #'user_mail': self.user_mail,
        }

        try:
            match detect(message):
                case 'de':
                    self.prompt = DE_PROMPT
                case 'en':
                    self.prompt = EN_PROMPT
                case _:
                    self.prompt = EN_PROMPT
        except:
            self.prompt = EN_PROMPT

        uuid = self.llm.submit_partial(self.prep_prompts([d])[0][0].text, stop = ["\nHuman:", " \n"])
        self.hash_table[uuid] = {
            'message': message,
            'reply': ""
        }
        return uuid

    def get_part(self, uuid, index):
        if uuid not in self.hash_table:
            return "[DONE]"

        try:
            text = self.llm.get_partial(uuid, index)
        except:
            return "[DONE]"

        if text == "[DONE]":
            if self.memory:
                self.memory.save_context(
                    inputs = {self.human_prefix: self.hash_table[uuid]['message'].strip()},
                    outputs = {self.ai_prefix: self.hash_table[uuid]['reply'].strip()}
                )
            if self.summary:
                self.summary.save_context(
                    inputs = {self.human_prefix: self.hash_table[uuid]['message'].strip()},
                    outputs = {self.ai_prefix: self.hash_table[uuid]['reply'].strip()}
                )
            if self.pghistory:
                self.pghistory.add_user_message(self.hash_table[uuid]['message'].strip())
                self.pghistory.add_ai_message(self.hash_table[uuid]['reply'].strip())

            del self.hash_table[uuid]

        elif text != "[BUSY]":
            self.hash_table[uuid]['reply'] += text

        return text

    def chat_history(self):
        ret = []
        if self.pghistory:
            for message in self.pghistory.messages:
                if type(message) == langchain.schema.HumanMessage:
                    ret.append({'Human': message.content})
                if type(message) == langchain.schema.AIMessage:
                    ret.append({'LibertyAI': message.content})
        return ret
