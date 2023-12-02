
from requests import get, post
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import json

from core.config import Config

from services.helpers.chat_tools import data_summary
# from services.integration.integration import remove_histor_image

from langchain.llms import OpenAI
from langchain.agents import Tool

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re

from services.helpers.index_engine import establish_private_dataset_index
from services.helpers.query_engine import private_dataset_query

MODEL_NAME = Config.MODEL_NAME
LLM = OpenAI(model_name=MODEL_NAME, temperature=0)  

API_KEY = "gqNlzI1bkwaMOcQe9SLBiClnO5atrpTM"
HEADER = {"x-dune-api-key" : API_KEY}

BASE_URL = "https://api.dune.com/api/v1/"

def make_api_url(module, action, ID):
    """
    We shall use this function to generate a URL to call the API.
    """

    url = BASE_URL + module + "/" + ID + "/" + action

    return url

def execute_query(query_id):
    """
    Takes in the query ID.
    Calls the API to execute the query.
    Returns the execution ID of the instance which is executing the query.
    """

    url = make_api_url("query", "execute", query_id)
    response = post(url, headers=HEADER)
    execution_id = response.json()['execution_id']

    return execution_id


def get_query_status(execution_id):
    """
    Takes in an execution ID.
    Fetches the status of query execution using the API
    Returns the status response object
    """

    url = make_api_url("execution", "status", execution_id)
    response = get(url, headers=HEADER)

    return response


def get_query_results(execution_id):
    """
    Takes in an execution ID.
    Fetches the results returned from the query using the API
    Returns the results response object
    """

    url = make_api_url("execution", "results", execution_id)
    response = get(url, headers=HEADER)

    return response


def cancel_query_execution(execution_id):
    """
    Takes in an execution ID.
    Cancels the ongoing execution of the query.
    Returns the response object.
    """

    url = make_api_url("execution", "cancel", execution_id)
    response = get(url, headers=HEADER)

    return response


# 查询去中心化交易所近7天以及近24小时交易量
query_id = "4319"

def get_data_from_dune(query_id):
    execution_id = execute_query(query_id)
    response = get_query_status(execution_id)

    result_metadata = response.json()
    print("response:", result_metadata)

    while result_metadata['state'] != 'QUERY_STATE_COMPLETED':
        time.sleep(1)  # 等待1秒钟
        response = get_query_status(execution_id)
        result_metadata = response.json()

    query_results = get_query_results(execution_id)
    data = pd.DataFrame(query_results.json()['result']['rows'])

    print("query results:", data)
    return data


# get_dex_volume(query_id)

# query total stablecoin supply
query_id = "14028"

# 查询去中心化交易所近7天以及近24小时交易量

import datetime
import secrets

# Function to generate a random string of specified length
def generate_random_string(length):
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    random_string = ''.join(secrets.choice(alphabet) for _ in range(length))
    return random_string


def dune_dex_exchange_volume(question):
    # remove_histor_image()

    query_id = "4319"

    data = get_data_from_dune(query_id)

    data = data.head(20)

    # Sort to ensure the chart displays in the order of Rank
    data = data.sort_values('Rank')

    # Set the size of the canvas
    plt.figure(figsize=(10, 6))

    # Create a bar chart for 24-hour trading volume
    plt.bar(data['Project'], data['24 Hours Volume'], alpha=0.8, label='24 Hours Volume')

    # Create a bar chart for 7-day trading volume
    plt.bar(data['Project'], data['7 Days Volume'], alpha=0.6, label='7 Days Volume')

    plt.xlabel('Project')  # Set x-axis label
    plt.ylabel('Volume')  # Set y-axis label
    plt.title('Volume of different projects')  # Set title
    plt.legend()  # Show legend

    plt.xticks(rotation=45)  # Rotate x-axis project names by 45 degrees to prevent overlap

    # Generate current time accurate to the hour
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H")

    # Generate a random string of length 8
    random_string = generate_random_string(8)

    # Include the current time and random string in the image name for differentiation
    image_filename = f"chart_{current_time}_{random_string}.png"
    image_fullpath = f"static/image/{image_filename}"

    # Save the chart to the specified file path and name
    plt.savefig(image_fullpath)

    # Display the chart
    plt.show()

    ans_dict = {
        "question": question,
        "data": data.to_dict(orient='records'),
        "image_link": image_fullpath
    }

    # print("---------ans dic",ans_dict)

    # Convert the dictionary to a JSON string
    ans_string = json.dumps(ans_dict)

    result = data_summary(ans_string)

    print("DEX VOLUME ANALYZE RESULT:", result)

    return result


# 查询去中心化交易所在不同链上的表现

def dune_weekly_dex_exchange_volume_by_chain(question):

    remove_histor_image()

    query_id = "2180075"

    data = get_data_from_dune(query_id)

    # Convert column '_col1' to datetime format
    data['_col1'] = pd.to_datetime(data['_col1'])

    # Sort the data by blockchain and date
    data = data.sort_values(['blockchain', '_col1'])

    # Group by blockchain and calculate the total volume
    total_volume_by_chain = data.groupby('blockchain')['usd_volume'].sum()

    # Select the top 20 chains with the highest total volume
    top_20_chains = total_volume_by_chain.nlargest(10).index

    # Only keep the data for the top 20 chains
    data_top_20 = data[data['blockchain'].isin(top_20_chains)]

    plt.figure(figsize=(12, 6))

    # Draw a line for each chain
    for chain in top_20_chains:
        chain_data = data_top_20[data_top_20['blockchain'] == chain]
        plt.plot(chain_data['_col1'], chain_data['usd_volume'], label=chain)

    plt.xlabel('Date')
    plt.ylabel('USD Volume')
    plt.title('Weekly DEX Exchange Volume by Chain')
    plt.legend()

    # Format the date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
    plt.gcf().autofmt_xdate()

    # Generate current time accurate to the hour
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H")

    # Generate a random string of length 8
    random_string = generate_random_string(8)

    # Include the current time and random string in the image name for differentiation
    image_filename = f"chart_{current_time}_{random_string}.png"
    image_fullpath = f"static/image/{image_filename}"

    plt.savefig(image_fullpath)

    plt.show()

    # Find the latest date in the data
    latest_date = data_top_20['_col1'].max()

    # Calculate the date four weeks ago
    four_weeks_ago = latest_date - pd.DateOffset(weeks=4)

    # Filter the data to only include the last four weeks
    data_last_4_weeks = data_top_20[data_top_20['_col1'] >= four_weeks_ago]

    ans_dict = {
        "question": question,
        "data": data_last_4_weeks.to_dict(orient='records'),
        "image_link": image_fullpath
    }

    # print("---------ans dic",ans_dict)

    # Convert the dictionary to a JSON string
    ans_string = json.dumps(ans_dict)

    result = data_summary(ans_string)

    print("DEX WEEKLY ANALYZE RESULT BY CHAIN:", result)

    return result


def dune_daily_dex_exchange_volume(question):

    query_id = "4388"

    data = get_data_from_dune(query_id)

    # Rename the columns for better readability
    data = data.rename(columns={"_col1": "Date", "project": "Project", "usd_volume": "USD Volume (Millions)"})

    # Convert column 'Date' to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Convert 'USD Volume (Millions)' from USD to millions of USD
    data['USD Volume (Millions)'] = data['USD Volume (Millions)'] / 1e6

    # Group by Project and Date, and calculate the sum of 'USD Volume (Millions)'
    data_grouped = data.groupby(['Project', 'Date']).sum().reset_index()

    # Select the top 15 projects with the highest total volume
    top_15_projects = data_grouped.groupby('Project')['USD Volume (Millions)'].sum().nlargest(15).index

    # Filter data to only include top 15 projects
    data_top_15 = data_grouped[data_grouped['Project'].isin(top_15_projects)]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw a line chart for each project
    for project in top_15_projects:
        project_data = data_top_15[data_top_15['Project'] == project]
        plt.plot(project_data['Date'], project_data['USD Volume (Millions)'], label=project)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # set the x-ticks to every day
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # format the x-ticks as YYYY-MM-DD
    fig.autofmt_xdate()  # autoformat the x-ticks for date

    plt.xlabel('Date')
    plt.ylabel('USD Volume (Millions)')
    plt.title('Daily DEX Exchange Volume by Project')
    plt.legend()

    # Generate current time accurate to the hour
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H")

    # Generate a random string of length 8
    random_string = generate_random_string(8)

    # Include the current time and random string in the image name for differentiation
    image_filename = f"chart_{current_time}_{random_string}.png"
    image_fullpath = f"static/image/{image_filename}"

    # Save the chart as a .png file
    plt.savefig(image_fullpath)

    plt.show()

    # Get the latest date in the data
    latest_date = data_top_15['Date'].max()

    # Get the date five days ago
    five_days_ago = latest_date - pd.DateOffset(days=5)

    # Filter the data to only include the last five days
    data_last_5_days = data_top_15[data_top_15['Date'] >= five_days_ago]

    # Group by project and calculate the sum of 'USD Volume (Millions)'
    data_last_5_days_grouped = data_last_5_days.groupby('Project')['USD Volume (Millions)'].sum().reset_index()

    # Sort the data by 'USD Volume (Millions)' in descending order
    data_last_5_days_grouped = data_last_5_days_grouped.sort_values(by='USD Volume (Millions)', ascending=False)

    # Convert DataFrame to string
    data_string = data_last_5_days_grouped.to_string(index=False)

    ans_dict = {
        "question": question,
        "data": data_string ,
        "image_link": image_fullpath
    }

    # print("---------ans dic",ans_dict)

    # Convert the dictionary to a JSON string
    ans_string = json.dumps(ans_dict)

    result = data_summary(ans_string)

    print("DEX DAILYLY ANALYZE RESULT:", result)

    return result



def stablecoin_supply_and_growth(question):
    query_id = "1652031"

    df = get_data_from_dune(query_id)
    
    # Convert the total_supply to billions
    df['total_supply'] = df['total_supply'] / 1e9

    # Exclude negative values
    df = df[df['total_supply'] > 0]

    # Sort df by total_supply and get the top 5
    top_4_df = df.sort_values('total_supply', ascending=False).head(4)

    # Remaining
    remaining_total_supply = df.loc[~df.index.isin(top_4_df.index), 'total_supply'].sum()
    remaining_df = pd.DataFrame([['Others', remaining_total_supply]], columns=['name', 'total_supply'])
    
    # Concatenate the top 5 df with the remaining df
    final_df = pd.concat([top_4_df, remaining_df])

    # Calculate the percentage
    final_df['percentage'] = final_df['total_supply'] / final_df['total_supply'].sum() * 100

    # Create a pie chart for the 'percentage' column
    fig, ax = plt.subplots(figsize=(16,9))
    wedges, texts, autotexts = ax.pie(final_df['percentage'], labels=final_df['name'], autopct=lambda p:f'{p:.1f}% ({p*sum(final_df["total_supply"])/100:.1f} billion)', wedgeprops=dict(width=0.3), pctdistance=0.85)
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title('Market Share of Different Stablecoins (Top 4 and Others)')

    # Generate current time accurate to the hour
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H")

    # Generate a random string of length 8
    random_string = generate_random_string(8)

    # Include the current time and random string in the image name for differentiation
    image_filename = f"chart_{current_time}_{random_string}.png"
    image_fullpath = f"static/image/{image_filename}"

    plt.savefig(image_fullpath)
    
    ans_dict = {
        "question": question,
        "data": final_df.to_dict(orient='records'),
        "image_link": image_fullpath
    }

    # print("---------ans dic",ans_dict)

    # Convert the dictionary to a JSON string
    ans_string = json.dumps(ans_dict)

    result = data_summary(ans_string)

    print("STABLECOIN SUPPLY AND GROWTH:", result)

    return result


# Set up the base template
STABLECOIN_AGENT_PROMPT_V2 = """Assume you're an investment research analyst with in-depth expertise in the areas of blockchain, web3, and artificial intelligence.
Now, your task is to use this knowledge to answer the upcoming questions in the most professional way. Before responding, carefully consider the purpose of each tool and whether it matches the question you need to answer.
You can use the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Action: the action to take, should be one of [{tool_names}]
Action Input: you should input {input} as the parameter of the selected tool

Begin! Choose the appropriate tool to obtain the answer and provide the observed result as your final response.

Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
         
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def stablecoin_agent(question):

    llm = LLM

    tools = [
        Tool(
            name="Stablecoin supply or growth",
            func=stablecoin_supply_and_growth,
            description="Useful for answering questions about stablecoin, such as 'What is the maket share about diferent stable coin?'"
        )

    ]

    prompt = CustomPromptTemplate(
        template=STABLECOIN_AGENT_PROMPT_V2,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()


    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )


    # agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    # result = agent.run(question)

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    result = agent_executor.run(question)
    

    print("INFO:     Agent Result:", result)

    return result


data_summary_prompt = '''
Assume that you are a research analyst specializing in investment. 
Your task is to describe the characteristics of the observed data in as much detail as possible. 
This includes identifying data trends, outliers, distribution features, and summarizing them meticulously. 
Since the data you typically deal with pertains to the web3 industry, 
you are also required to provide investment advice based on the characteristics of the data. 
Please list your insights and recommendations individually and as detailed as possible, 
while maintaining a high level of professionalism in your summary
'''

# set prompt

# Set up the base template
DEX_AGENT_PROMPT_V1 = """ Assume you're a master of on-chain data analysis. 
Your task is to understand the type and scope of data the user is seeking, based on their input. 
Choose the most suitable tools to retrieve this data from the blockchain. Upon obtaining the data, 
you're required to synthesize it in response to the user's original question.if you find one or more image links, the answer must contains them,they are come from data

Here is the flow you should follow:
1. Interpretation: Understand the user's query and identify what type of data and which aspects they are interested in.
2. Selection: Choose the best tools from your repertoire to retrieve the desired data.
3. Execution: Use these tools to obtain the necessary data from the blockchain.
4. Response: Analyze the obtained data and formulate a comprehensive and understandable answer that directly addresses the user's original query.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: you should input {input} as param of tool
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times), 
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""


# Set up the base template
DEX_AGENT_PROMPT_V2 = """Assume you're an investment research analyst with in-depth expertise in the areas of blockchain, web3, and artificial intelligence.
Now, your task is to use this knowledge to answer the upcoming questions in the most professional way. Before responding, carefully consider the purpose of each tool and whether it matches the question you need to answer.
You can use the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Action: the action to take, should be one of [{tool_names}]
Action Input: you should input {input} as the parameter of the selected tool

Begin! Choose the appropriate tool to obtain the answer and provide the observed result as your final response.

Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
         
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def dex_agent(question):

    llm = LLM

    tools = [
        Tool(
            name="dex trading volume or market share",
            func=dune_dex_exchange_volume,
            description="Useful for answering questions about decentralized exchanges, such as 'What is the trading volume of dex 24 in the past 24 hours?'"
        ),
        Tool(
            name="DEX trading volume by chain weekly",
            func=dune_weekly_dex_exchange_volume_by_chain,
            description="Useful for answering questions about trading volume of decentralized exchanges on different chains."
        ),
        Tool(
            name="DEX trading volume daily",
            func=dune_daily_dex_exchange_volume,
            description="Useful for answering questions about trading volume of decentralized exchanges daily."
        )

    ]

    prompt = CustomPromptTemplate(
        template=DEX_AGENT_PROMPT_V2,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()


    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )


    # agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    # result = agent.run(question)

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    result = agent_executor.run(question)
    

    print("INFO:     Agent Result:", result)

    return result

def remove_histor_image():

     # 先删除历史图片
    image_path = f'static/image/chart.png'

    # 如果文件存在，删除它
    if os.path.exists(image_path):
        os.remove(image_path)
        # print("文件已删除")
    # else:
        # print("文件不存在:",image_path)