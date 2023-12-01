"""
This is an attempt to create a polite chatbot.
Currently, there are a few bugs with it: 
- It keeps trying to rephrase the same sentence over and over again.
- It tries to rephrase the input instead of its own output.
"""
from dotenv import load_dotenv
from langchain.chains.conversation.memory import \
    ConversationSummaryBufferMemory

load_dotenv()
import os

from langchain.agents import initialize_agent, tool
from langchain.llms import OpenAI, OpenAIChat

# ChatGPT
chatllm = OpenAIChat()
# GPT-3.5
llm = OpenAI()


@tool("check_sentiment")
def check_sentiment(text) -> str:
    """Attitude check."""
    return chatllm(
        f"What is the tone of this sentence? Summarize with one word:\n{text}"
    )


@tool("rephrase")
def filter(text) -> str:
    """Rephrase for politeness."""
    return chatllm(
        f"Rephrase the following sentence with politeness and curiosity: {text}"
    )


agent = initialize_agent(
    [
        filter,
        check_sentiment,
    ],
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=70,
    ),
    agent_kwargs={
        "prefix": "You are a friendly chatbot. Answers your responses to humans with humbleness. First try to answer the question and if it needs rephrasing, do so and answer with the rephrase.",
    },
)

while True:
    text = input("You: ")
    print("Bot:", agent(f"Human: {text}")["output"])
