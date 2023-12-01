# Jak użyć Langchain do interakcji z innym oprogramowaniem (Langchain agent + Langchain tools),

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory



if __name__ == '__main__':
    with open('api.key', 'r') as openai_api_key:
        openai_api_key = openai_api_key.read().strip()

    wikipedia = WikipediaAPIWrapper()

    llm = OpenAI(
        model_name="text-davinci-003",
        openai_api_key=openai_api_key
    )

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    tools = [
        Tool.from_function(
            func=wikipedia.run,
            name = "Wikipedia",
            description="Useful when you need to find a Wikipedia articles."
        ),
    ]

    agent = initialize_agent(
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )

    print(agent('Find an article about Nile. What is the length of the river?'))

    print(agent('The river flows into a sea. What is the area of the sea?'))


# {'input': 'The river flows into a sea. What is the area of the sea?', 'chat_history': [HumanMessage(lc_kwargs={'content': 'Find an article about Nile. What is the length of the river?'}, content='Find an article about Nile. What is the length of the river?', additional_kwargs={}, example=False), AIMessage(lc_kwargs={'content': 'According to the article, the Nile is approximately 6,650 km (4,130 mi) long.'}, content='According to the article, the Nile is approximately 6,650 km (4,130 mi) long.', additional_kwargs={}, example=False)], 'output': 'The Mediterranean Sea covers an area of approximately 2,500,000 km2 (970,000 sq mi).'}