from langchain.chains import ConversationalRetrievalChain
from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from lazygitgpt.llms import chat_model
from lazygitgpt.datasources.repos import read_repository_contents
from lazygitgpt.git.operations import update_files
from lazygitgpt.retrievers.retrievalqa import retriever
from lazygitgpt.memory.memory import memory

search = DuckDuckGoSearchRun()

def generate_response(prompt):
    inputs = {'chat_history': '', 'question': prompt}
    qa = ConversationalRetrievalChain.from_llm(chat_model, retriever=retriever, memory=memory)
    result = qa(inputs)
    return result["answer"]

# tools = [
#     Tool(
#         name='DuckDuckGo Search',
#         func= search.run,
#         description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
#     ),
#     Tool(
#         name='Conversational Retrieval',
#         func=generate_response,
#         description="This is Conversational Retrieval chain which has content of the entire repository."
#     )
# ]

# zero_shot_agent = initialize_agent(
#     agent="zero-shot-react-description",
#     tools=tools,
#     llm=chat_model,
#     verbose=True,
#     max_iterations=30,
#     retriever=retriever
# )

# def run(prompt):
#     reponse = zero_shot_agent.run(prompt)
#     return reponse
