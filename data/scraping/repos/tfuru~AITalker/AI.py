import re
import logging
from typing import Any

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.agents import Tool, initialize_agent
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.cache import SQLiteCache
from langchain.callbacks.manager import CallbackManager, BaseCallbackHandler

# 参考
# https://note.com/strictlyes/n/n6de1a36a6e7e

# langchain キャッシュの無効化
# langchain.llm_cache = None

# SQLiteCache によるキャッシュの有効化
langchain.llm_cache = SQLiteCache(database_path="ai/cache/.langchain.db")

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, callback):
        self.callback = callback
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self.callback is not None:
            self.callback(token) 

class AI:
    def __init__(self):
        # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
        temperature=0.0
        # gpt-3.5-turbo-16k
        self.llm=ChatOpenAI(
            streaming=True,
            callback_manager=CallbackManager([StreamingCallbackHandler(self.callback_streaming)]),
            temperature=temperature,
            model_name='gpt-3.5-turbo'
            )
        
        # プロンプト作成
        system_template="""あなたは音声で対話するAIチャットボットです。
        小学生にも分かるように簡単に短く会話するようにしてください。
        語尾に必ず にゃん をつけてください。

        過去の会話履歴を参考にして会話をしてください。
        [会話履歴]
        {chat_history}
        """
        human_template="{input}"
        self.prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template),
            ])
        self.memory=ConversationBufferWindowMemory(
            k=5, # 記憶回数
            memory_key="chat_history",
            human_prefix="User", 
            ai_prefix="Ai",
            input_key="input",
            return_messages=True
            )
    
    def callback_streaming(self, token):
        if self.answer_callback is None:
            return
        
        # logging.info(f"streaming token: {token}")
        self.answer += f'{token}'
        result=re.search(r'[\n!,。、！？]', self.answer)
        if result is None:
            return
        logging.info(f"streaming answer: {self.answer}")
        # 終端記号が出現したらコールバックを呼び出す
        self.answer_callback(self.answer)
        self.answer=''
    
    def request(self, input, answer_callback=None):
        self.answer=''
        self.answer_callback = answer_callback

        # verbose プロンプト途中結果の表示有無
        chain=LLMChain(llm=self.llm, 
                          prompt=self.prompt,
                          memory=self.memory,
                          verbose=False)
        result=chain.run(input=input)
        self.answer=result.strip()
        return self.answer
    
    # TODO Agent と tools を使った会話 だが、現状は想定通りに動作しない
    def request_agent(self, text):
        search = GoogleSearchAPIWrapper()
        tools = [
            Tool(
                name = "Search",
                func=search.run,
                description="Helpful if you need to answer a question"
            )
        ]
        prefix = """Please answer the following questions in Japanese as briefly as possible. You can access the following tools:"""
        suffix = """Don't forget to write your final answer in Japanese. Please be sure to add "nyan" at the end."""
        agent = initialize_agent(
            tools,
            self.llm,
            # agent="chat-zero-shot-react-description",
            agent="chat-conversational-react-description",
            verbose=True,
            memory=self.memory,
            prefix=prefix,
            suffix=suffix)
        
        try:
            result = agent.run(input=text)
        except ValueError as e:
            result = str(e)
            if not result.startswith("Could not parse LLM output: `"):
                raise e
            result = result.removeprefix("Could not parse LLM output: `").removesuffix("`")

        answer=result.strip()
        logging.debug(answer)
        # buffer = self.memory.load_memory_variables({})
        # logging.info(f'memory buffer {buffer}')
        return answer
