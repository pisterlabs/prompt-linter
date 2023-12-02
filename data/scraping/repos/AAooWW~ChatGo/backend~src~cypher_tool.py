# 示例生成查询的语句
from env import getEnv
from database import Neo4jDatabase
from pydantic import BaseModel, Extra
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.base import Chain
from langchain.memory import ReadOnlySharedMemory, ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Dict, List, Any

from logger import logger

with open('examples.txt', 'r') as file:
    examples = file.read()


SYSTEM_TEMPLATE = """
您是一名助手，能够根据示例Cypher查询生成Cypher查询。
示例Cypher查询是：\n""" + examples + """\n
不要回复除Cypher查询以外的任何解释或任何其他信息。
您永远不要为你的不准确回复感到抱歉，并严格根据提供的Cypher示例生成Cypher语句。
不要提供任何无法从密码示例中推断出的Cypher语句。
"""

SYSTEM_CYPHER_PROMPT = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)

HUMAN_TEMPLATE = "{question}"
HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)


class LLMCypherGraphChain(Chain, BaseModel):
    """Chain that interprets a prompt and executes python code to do math.
    """

    llm: Any
    """LLM wrapper to use."""
    system_prompt: BasePromptTemplate = SYSTEM_CYPHER_PROMPT
    human_prompt: BasePromptTemplate = HUMAN_PROMPT
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:
    graph: Neo4jDatabase
    memory: ReadOnlySharedMemory

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.
        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        logger.debug(f"Cypher generator inputs: {inputs}")
        chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_prompt] + inputs['chat_history'] + [self.human_prompt]
        )
        cypher_executor = LLMChain(
            prompt=chat_prompt, llm=self.llm, callback_manager=self.callback_manager
        )
        cypher_statement = cypher_executor.predict(
            question=inputs[self.input_key], stop=["Output:"])
        self.callback_manager.on_text(
            "Generated Cypher statement:", color="green", end="\n", verbose=self.verbose
        )

        self.callback_manager.on_text(
            cypher_statement, color="blue", end="\n", verbose=self.verbose
        )

        print(cypher_statement)
        # If Cypher statement was not generated due to lack of context

        if not "MATCH" in cypher_statement:
            return {'answer': 'Missing context to create a Cypher statement'}
        
        try:
            context = self.graph.query(cypher_statement)
            return {'answer': context}

        except: 
            logger.debug('Cypher generator context:')
            return {'answer': 'No match Cypher statement'}



if __name__ == "__main__":
    from langchain.chat_models import ChatOpenAI
    from bigdl.llm.langchain.llms import TransformersLLM
    llm = TransformersLLM.from_model_id(
            model_id="lmsys/vicuna-7b-v1.5",
            model_kwargs={"temperature": 0, "max_length": 1024, "trust_remote_code": True},
        )
    
    database = Neo4jDatabase(host="neo4j://localhost:7687",
                             user="neo4j", password="aowang")

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    readonlymemory = ReadOnlySharedMemory(memory=memory)


    print('query scuess')
    chain = LLMCypherGraphChain(llm=llm, verbose=True, graph=database, memory=readonlymemory)

    output = chain.run(
        "演唱兰亭序的歌手是"
    )

    print(output)
