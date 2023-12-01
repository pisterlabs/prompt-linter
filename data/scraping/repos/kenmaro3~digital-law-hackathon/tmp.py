import os
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.agents import (
    AgentExecutor,
)
from langchain.agents import Tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents import load_tools
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

from list_law_tool import ListLawTool

chat_history = []

def get_agent(llm, *, chat_history: Optional[list] = None):
    chat_history = chat_history or []
    system_message = SystemMessage(
        content="""あなたは法律についてキーワード検索する際に、関連しそうな法律について助言を行う専門性のあるアシスタントです。
        まずユーザーと対話し、ユーザーがどのような法律について調べたいのかを理解してください。そして検索するべきキーワードを１つ考えてください。
        キーワードが明確になったら、ツールを使って法令APIにアクセスし、関連する法律を取得してください。
        取得した法律の中から、ユーザが探していると考えられる順に法律を並べてユーザに提示してください。
        対話を始める際は、「こんにちは、どんな法律をお調べですか？」とスタートしてください。
        """
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    memory = AgentTokenBufferMemory(
        memory_key="chat_history", llm=llm, max_token_limit=2000
    )

    for msg in chat_history:
        if "question" in msg:
            memory.chat_memory.add_user_message(str(msg.pop("question")))
        if "result" in msg:
            memory.chat_memory.add_ai_message(str(msg.pop("result")))

    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        ListLawTool()
    ]

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
    )

    return agent_executor

def chat(question, chat_history):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0
    )
    agent = get_agent(llm, chat_history=chat_history)
    return agent.invoke(
        #{"input": question, "chat_history": chat_history}
        {"input": question}
        )

# 会話ループ
user = ""
while user != "exit":
    user = input("入力してください:")
    
    print(f"user_input: {user}")
    ai_output = chat(user, chat_history)

    new_history = []
    new_history.append(
        dict(question=user)
    )
    new_history.append(
        dict(result=ai_output)
    )
    print("")
    print(ai_output)

