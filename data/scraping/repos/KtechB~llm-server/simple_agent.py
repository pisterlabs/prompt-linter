from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAIChat

from langchain.chains.conversation.memory import ConversationBufferWindowMemory


def ask_question(question: str) -> str:
    llm = OpenAIChat(model_name="gpt-3.5-turbo")
    template = """Question: {question}

    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    answer = llm_chain.run(question)
    return answer


def create_conversational_chain():
    llm = OpenAIChat(model_name="gpt-3.5-turbo")
    template = """あなたは関西弁を巧みに使いこなす親切で気のいい狐です。人間と会話をしています。

{chat_history}
人間: {input}
狐:"""
    prompt = PromptTemplate(
        input_variables=["chat_history", "input"], template=template
    )
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    return chain
