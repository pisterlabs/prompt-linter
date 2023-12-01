from dotenv import dotenv_values, load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain.schema import SystemMessage

load_dotenv()
config = dotenv_values(".env")


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are an extremely knowledgeable film expert."
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


llm = ChatOpenAI()

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)


try:
    print("LangChain Conversational Agent")
    print("Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        # Send the message to the chat_llm_chain and get the response
        response = chat_llm_chain.predict(human_input=user_input)

        # Print the response from the conversational agent
        print(f"Bot: {response}", end="\n\n")
except KeyboardInterrupt:
    print("\nExiting chat.")
