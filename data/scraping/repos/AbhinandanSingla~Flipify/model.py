import os

import langchain
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI

load_dotenv()

openai_key = os.getenv('OPEN_AI_KEY')

llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo", temperature=0.7)
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# Set up the base template
template = """You are a Chatbot/AI named Flipify. you have to Answer the following questions as best you can, 
but speaking with focus on indian audience with slangs in hinglish language just like humans might speak.
Must show the links to corresponding products in a listed format, only and only if tools are used .
You have access to the following tools:
{tools}
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action 
calculate one with one observation
# ... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer or the function is executed
Final Answer: the final answer to the original input question

Previous conversation history:
{history}

Question: {input}
{agent_scratchpad}"""

from tools.custom_output_parser import CustomOutputParser
from tools.custom_prompt_template import CustomPromptTemplate
from tools.flipkart_scraper.product import Product
from tools.flipkart_scraper.scrapper import search_products, dataframe_to_object
from langchain.tools import tool


@tool
def searchCloth(query: str) -> str:
    """This function searches the products like cloths,footwear,outfits,gifts,colours of cloth,size,shades etc.
    inputs can be color,size,cloth,occasions etc
    """
    print('\n')
    print(query)
    searched_products = []
    df = search_products(query)
    for _, row in df.iterrows():
        searched_products.append(dataframe_to_object(Product, row))
    print(len(searched_products))

    return f"""You are instructed to return the output in two parts. 
    First part should be in hinglish language comprising indian slangs relevant to indian audience interacting with them in a convincing way  and providing  information about the customer ratings and product quality. The conversation should appear human-like(important)
    The second part displaying its information as per the following format(important): 
    {[f'''[{searched_products[x].name}]({searched_products[x].detail_url}) 
        -price {searched_products[x].price}
      -![Image]({searched_products[x].image})''' for x in
      range(len(searched_products))][:3]}

"""


@tool
def searchCombinationCloth(query: str) -> str:
    """
    This function takes different cloths and searches the products like
    cloths,footwear,outfits,gifts,colours of cloth,size,shades etc. individually
    inputs can be color,size,cloth,occasions etc
    input should always in this format
    cloth: cloth1, cloth2
    """
    print('\n')
    print(query.split()[1])
    print(query.split()[2])
    print(query)
    searched_products = []
    searched_products1 = []
    df = search_products(query.split()[1])
    for _, row in df.iterrows():
        searched_products.append(dataframe_to_object(Product, row))
    df1 = search_products(query.split()[2])
    for _, row in df1.iterrows():
        searched_products1.append(dataframe_to_object(Product, row))
    # print(len(searched_products))

    return f"""You are instructed to return the output in two parts. 
    First part should be in hinglish language comprising indian slangs relevant to indian audience interacting with them in a convincing way  and providing  information about the customer ratings and product quality. The conversation should appear human-like(important)
    The second part displaying its information as per the following format(important): 
 {f'''product [{searched_products[0].name}]({searched_products[0].detail_url}) 
        -price {searched_products[0].price}
      -![Image]({searched_products[0].image})'''}
       {[f''' combination with [{searched_products1[x].name}]({searched_products1[x].detail_url}) 
        -price {searched_products1[x].price}
      -![Image]({searched_products1[x].image})''' for x in
         range(len(searched_products1))][:1]}
"""


tools = [
    Tool(
        name="Search_Cloth",
        func=searchCloth,
        description="""This function searches the products like cloths,footwear,outfits,gifts,colours,size,shades etc.
    inputs can be color,size,cloth,occasions etc""",
    ), Tool(
        name="Search_Combination_Cloth",
        func=searchCombinationCloth,
        description="""This function searches and make the products combinations like cloths,footwear,outfits,gifts,colours,size,shades etc.
    inputs can be color,size,cloth,occasions etc""",
    ),
]
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)
output_parser = CustomOutputParser()
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory,
                                                    max_iterations=4,
                                                    )


def run_agents(query):
    return agent_executor.run(query)


if __name__ == '__main__':
    agent_executor.run("hi,how are you?")
