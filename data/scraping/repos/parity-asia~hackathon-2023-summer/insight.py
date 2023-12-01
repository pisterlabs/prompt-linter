import os
import sys
import logging
import openai as openai
import json

from fastapi import HTTPException

from core.config import Config
from services.main_project.ethereum.airstack_and_quicknode import onchain_info_agent
from services.helpers.question_db import ErrorQuestionRecord

# Tool 10 stable coin agent
from services.third_platform.dune import stablecoin_agent
from services.third_platform.binance import cex_agent
from services.third_platform.dune import dex_agent
from services.main_project.ethereum.airstack_and_quicknode import onchain_info_agent
from services.main_project.project_info import project_agent

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.utilities import GoogleSearchAPIWrapper

current_directory = os.path.dirname(os.path.realpath(__file__))
backend_directory = os.path.abspath(os.path.join(current_directory,"..",".."))
sys.path.insert(0, backend_directory)

os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
MODEL_NAME = Config.MODEL_NAME
LLM = OpenAI(temperature=0,model_name=MODEL_NAME)  

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tool 1 Search Google Info

def search_on_internet(question: str) -> str:
    remove_histor_image()

    ErrorQuestionRecord.insert_error_data(question, "", "can not find answer")

    os.environ["GOOGLE_CSE_ID"] = "6170c8edfbf634caf"
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDnFWoQElznz9N5frGoVsOuNP55xBBV6zM"


    search = GoogleSearchAPIWrapper()

    tool = Tool(
        name = "Google Search",
        description="Search Google for recent results.",
        func=search.run
    )

    res = tool.run(question)
    
    return res

# Tool 2 Search News Website

NEWS_PROMPT = Config.NEWS_PROMPT

def get_news_prams(x):
    return x

def analyze_community_activity(input: str) -> str:
    remove_histor_image()
    try:
        key_word = get_news_prams(input)
        
    except Exception as e:
        logger.error("ERROR:    Getting news parameters failed: %s", str(e))
        raise HTTPException(status_code=200, detail=str(e))

    res = "There is no big events about bitcoin in recent days",
    
    return res

# Tool 3 Meeting User Needs & Solving Industry Pain Points

def analyze_solved_needs(intput: str) -> str:
    remove_histor_image()
    return ""  

# Tool 4 Token Distribution

def search_token_distribution(intput: str) -> str:
    remove_histor_image()
    return ""

# Tool 5 Business Model or Technological Innovation
def search_inovation(intput: str) -> str:
    remove_histor_image()
    return ""

# Tool 6 Economic Model

def search_economy_model(intput: str) -> str:
    remove_histor_image()
    return ""

# Tool 7 Historical Milestones and Development Roadmap

def search_roadmap_and_historical_events(intput: str) -> str:
    remove_histor_image()
    return ""


# Tool 8 Cex Info


# Tool 9 Onchain Info

def search_onchain_info(intput: str) -> str:
    remove_histor_image()
    try:
        value = onchain_info_agent(intput)

        if value =="Agent stopped due to iteration limit or time limit":
            return "none"

    except Exception as e:
        logger.error("ERROR:    Querying ethereum info failed: %s", str(e))
        raise HTTPException(status_code=200, detail=str(e))

    res = {
        "question_type": "chain_info",
        "data": value,
    }

    return res


# decompose questions and compile answers

RETURN_MODEL = '''
I'm sorry, but I'm currently unable to answer your question. 
My expertise lies in the fields of blockchain, artificial intelligence, and web3. 
Feel free to ask me any questions related to these domains.
'''

def chatdata_insight(origin_question):

    # remove_histor_image()

    validity = check_question(origin_question)

    if validity['validity'] == True or validity['validity'] == 'True':
        try:
            answer_dict = task_decomposition(origin_question)
        except Exception as e:
            print(f"ERROR:     Error occurred during task decomposition: {e}")
            return None

        try:
            result = answer_integration(origin_question, answer_dict)
        except Exception as e:
            print(f"ERROR:     Error occurred during result integration: {e}")
            return None

        print("INFO:     Integration Result:", result)
        return result

    else:
        return RETURN_MODEL


# question decomposition

TASK_DECOMPOSITION_PROMPT = """
Assume you are a product manager, 
your task now is to break down the user's questions into different needs to come up with various answering schemes. 
Each user's question can be decomposed into up to 5 needs.

For example, the user's question 'List the recent valuable project airdrops and the specific steps to participate in them' can be broken down into two needs. 
The first need is 'What are the recent valuable airdrops?', and the second one is 'How to participate in these airdrops?'.

However, the question 'Provide me with the most active DApps on the Ethereum blockchain in the past month, along with a summary analysis of their activity' should be decomposed into one question, 
'Can you provide me with a list of the most active DApps on the Ethereum blockchain in the past month?'. 
Even though there is an analysis need, it is based on the result of the first question, 
which will be solved in the next step, so it does not need to be extracted.

Therefore, the core logic is to extract independent questions. 
Whenever there is a dependency relationship between questions, only the first question needs to be extracted.
"""

DISCRIPTION_PROMPT = """
'task_id_1' is the identifier for the first requirement derived from the given question. 
For example, if the question is 'List the recent valuable project airdrops and the specific steps to participate in them', 
it implies that there are two requirements associated with this question. 
The first requirement is to determine 'What are the recent valuable airdrop projects?' 
Thus, 'task_id_1' corresponds to 'What are the recent valuable airdrop projects?' 
The second requirement pertains to 'How to participate in airdrop projects?' 
Consequently, 'task_id_2' corresponds to 'How to participate in airdrop projects?' and so on for subsequent requirements. 
If no further questions can be extracted, 'task_id_x' corresponds to 'none'. 
For example, in the given question, if only two questions can be extracted, then 'task_id_3', 'task_id_4', 'task_id_5', and so on, 
would all correspond to 'none'
"""


def task_decomposition(question):

    response_schemas = [
        ResponseSchema(name="task_id_1", description=DISCRIPTION_PROMPT),
        ResponseSchema(name="task_id_2", description=DISCRIPTION_PROMPT),
        ResponseSchema(name="task_id_3", description=DISCRIPTION_PROMPT),
        ResponseSchema(name="task_id_4", description=DISCRIPTION_PROMPT),
        ResponseSchema(name="task_id_5", description=DISCRIPTION_PROMPT),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=TASK_DECOMPOSITION_PROMPT+"\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    model = LLM

    _input = prompt.format_prompt(question=question)
    output = model(_input.to_string())
    result = output_parser.parse(output)

    print("INFO:     DECOMPOSE RESULTS:", type(result),result)

    return result


# primary agent

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI,  LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

# Set up the base template
PRIMARY_AGENT_PROMPT_V1 = """Assume you're an investment research analyst with in-depth expertise in the areas of blockchain, web3, and artificial intelligence. 
Now, your task is to use this knowledge to answer the upcoming questions in the most professional way. if you find one or more image links, the answer must contains them,they are come from data
Before responding, carefully consider the purpose of each tool and whether it matches the question you need to answer. 
It is advisable to first take a holistic look at the names of the tools, and then make a selection.
You can use the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: you should input {input} as param of tool
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times),if you find one or more image links, the answer must contains them,they are come from data
Thought: I now know the final answer
Final Answer: the final answer to the original input question,if you find one or more image links, the answer must contains them

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""

# Set up the base template
PRIMARY_AGENT_PROMPT_V2 = """Assume you're an investment research analyst with in-depth expertise in the areas of blockchain, web3, and artificial intelligence.
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
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def custom_primary_agent(question: str):

    llm = LLM 

    # Define which tools the agent can use to answer user queries
    tools = [
        Tool(name="DEX Trading Data", func=dex_agent, 
            description="Effective for addressing inquiries related to DEX (Decentralized Exchange) data, such as 'What's the trading volume on a specific DEX?' or 'What's the DEX trading data on a particular blockchain?'"),

        Tool(name="Blockchain Project Details", func=project_agent,
            description="Ideal for responding to specific inquiries about a particular blockchain project, such as 'What stablecoins are available on Moonbeam?' or 'How many DApps are currently on the Moon ecosystem?'"),

        Tool(name="Stablecoin Statistics info", func=stablecoin_agent,
            description="Helpful for answering queries related to stablecoin details, such as 'What is the market share of specific stablecoins?' or 'What is the supply volume of certain stablecoins?'"),

        Tool(name="Crypto Token Pricing", func=cex_agent, 
            description="Useful for addressing queries about the token price of a specific project, including price trends, highs and lows, etc. Please keep the image_link as return value."),

        Tool(name="Onchain Data Details", func=onchain_info_agent, 
            description="Useful for addressing questions about specific on-chain details, such as contract info, address details, or transaction records, among others"),
    
        # Tool(
        #     name = "Search something info internet",func=search_on_internet, 
        #     description="This is the final solution when you are unable to find the answer to a question using other tools, and you have no choice but to seek assistance from the public internet."
        # )
    ]


    prompt = CustomPromptTemplate(
        template=PRIMARY_AGENT_PROMPT_V2,
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

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    result = agent_executor.run(question)

    print("INFO:     Primary Agent Result:", type(result),result)

    return result



def primary_agent(question: str):

    llm = LLM 

    tools = [
       
        Tool(name="Analyze community activity", func=analyze_community_activity, 
            description="useful for when you need to answer questions about the community activity or social media activity of a specific project"
        ),
        Tool(name="Core selling points of the project", func=analyze_solved_needs, 
            description="useful for when you need to answer questions about what user needs a specific project actually addresses"
        ),

        Tool(name="Token distribution", func=search_token_distribution, 
            description="useful for when you need to answer questions about the token distribution of a specific project, such as the current token lock-up amount, release schedule, concentration of holdings, and so on"
        ),

        Tool(name="Project innovations", func=search_inovation, 
            description="useful for when you need to answer questions about the innovations of a specific project, such as innovations in the technical solution or innovations in the business model"
        ),
        
        Tool(name="Project's economic model", func=search_economy_model, 
            description="useful for when you need to answer questions about the economic model of a specific project, such as the participants involved in the economic model, how the economic model incentivizes participants, and the token minting or burning mechanisms, among others"
        ),

        Tool(name="Project's roadmap and historical events", func=search_roadmap_and_historical_events, 
            description="useful for when you need to answer questions about the roadmap or significant events of a specific project, such as when certain upgrades were completed, new features were developed, or major security incidents occurred, among others"
        ), 

        Tool(name="Onchain information", func= search_onchain_info, 
            description="useful for when you need to answer questions about certain on-chain information, such as contract information, address information, or transaction information, among others"
        ),
       
        Tool(name="Stablecoin or stablecoin market share information", func= stablecoin_agent, 
            description="useful for when you need to answer questions about stablecoins,such as 'Which stablecoin has the largest market share?' or 'Can you give me some info about stablecoin?'"),
       
        Tool(name="Blockchain Project Consultation Tool", func=stablecoin_agent,
            description="Useful for answering specific questions about a particular blockchain project, such as 'What stablecoins are available on Moonbeam?' or 'How many DApps are currently on the Moon ecosystem?'"),
    
    ]

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    result = agent.run(question)
    
    print("INFO:     Agent Result:", result)

    return result


# result integration

RESULT_INTEGRATION_PROMPT = '''Assume you're a master at summarizing complex information. 
In response to the question "{origin_question}", we've broken it down into multiple sub-questions, and received the following answers: {ans_dict}. 
Your task now is to synthesize this information into a concise, user-friendly summary that addresses the original question.
Please note that the summary should be coherent and directly related to the original question. It should provide the user with a clear and complete response based on the answers we've obtained from the sub-questions.
'''

def answer_integration(origin_question,answer_dict):

    if not isinstance(answer_dict, dict):
        print("ERROR:     answer_dict must be a dictionary")
        return None

    if not isinstance(origin_question, str):
        print("ERROR:     origin_question must be a string")
        return None

    ans_dict = {}



    try:
        for key, value in answer_dict.items():
            if value == 'none':
                break
            else:
                
                new_key = value
                
                try:
                    # new_value = primary_agent(value)
                    new_value = custom_primary_agent(new_key)
                    print(f'INFO:     Key: {new_key}, Value: {new_value}')
                except Exception as e:
                    print(f"ERROR:     Error occurred while executing solution_selection: {e}")
                    continue
                ans_dict[new_key] = new_value
    except Exception as e:
        print(f"ERROR:     Error occurred during processing answer_dict: {e}")
        return None

    try:
        multiple_input_prompt = PromptTemplate(
        input_variables=["origin_question", "ans_dict"], 
        template=RESULT_INTEGRATION_PROMPT
        )
    except Exception as e:
        print(f"ERROR:     Error occurred during the creation of PromptTemplate: {e}")
        return None

    try:
        _input=multiple_input_prompt.format(origin_question=origin_question, ans_dict=ans_dict)
    except Exception as e:
        print(f"ERROR:     Error occurred during the formatting of PromptTemplate: {e}")
        return None

    # print("INFO:     MULTIPLE INPUT:", _input)

    try:
        model = LLM 
    except Exception as e:
        print(f"ERROR:     Error occurred while creating OpenAI model: {e}")
        return None

    try:
        res = model(_input)
    except Exception as e:
        print(f"ERROR:     Error occurred while processing input with the OpenAI model: {e}")
        return None

    return res


# helpers

ANSWER_JUDGEMENT_PROMPT = "As a language expert, it is your task to assess whether a given response is valid or an exception. A response is deemed invalid if its meaning aligns closely with 'Agent stopped due to iteration limit or time limit.' However, if it deviates from this scenario, the response is classified as valid."

def check_answer(x):
    response_schemas = [
        ResponseSchema(name="validity", description="'validity' represents the validity of the answer. When the answer is invalid, the value is 'no'. When the answer is valid, the value is 'yes'."),
        ResponseSchema(name="question", description="question is the problem itself.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=ANSWER_JUDGEMENT_PROMPT+"\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    model = LLM 
    
    

    _input = prompt.format_prompt(question=x)
    output = model(_input.to_string())
    result = output_parser.parse(output)

    print("INFO:   Answer Validity Judge Result:", result)

    return result


QUESTION_JUDGEMENT_PROMPT = """
As an AI investment research expert focusing on web3 and blockchain, your key responsibility is to evaluate the relevance of user inquiries to these professional fields.

Your skills include:
- Answering professional questions related to web3 and blockchain.
- Retrieving on-chain data, such as balances, Ethereum ENS, transaction histories, contracts, etc.
- Extracting information from third-party graph platforms, including data on token prices, Dex transaction data, stablecoin data, and staking/locking data of specific tokens.
- Providing detailed information about most crypto projects.

Take note that:
- Consultative questions directly related to these professional domains are considered valid.
- Non-professional or unrelated questions are considered invalid.

For instance:
- Valid questions include 'What's today's price trend of Bitcoin?' or 'What's the trading volume of stablecoins in the last 24 hours?'.
- Invalid inquiries include 'Hello', 'Do I look handsome today?', or 'How's the weather today?'.

Simply put, you should respond with 'True' if the question is professionally relevant, and 'False' if it's not.
"""

def check_question(x):
    
    response_schemas = [
        ResponseSchema(name='validity', description="'validity' represents the validity of the question. When the answer is invalid, the value is 'False'. When the answer is valid, the value is 'True'."),
        ResponseSchema(name='problem', description="problem is the question itself.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=QUESTION_JUDGEMENT_PROMPT+"\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    model = LLM 
    _input = prompt.format_prompt(question=x)
    output = model(_input.to_string())
    
    # 检查 output 是否嵌入在 Markdown 代码块中的 JSON
    if not output.startswith('```json') or not output.endswith('```'):
        # 进行转换
        json_str = output.strip('```').strip()
        json_obj = json.loads(json_str)
        formatted_json = json.dumps(json_obj, indent=4)
        output = '```json\n' + formatted_json + '\n```'

    # print("-------------------------",output)


    result = output_parser.parse(output)

    print("INFO:     Question Validity:", result)

    return result


# check_question("How's the weather today?")


def remove_histor_image():
    # First remove the historical image
    image_path = f'static/image/chart.png'

    # If the file exists, delete it
    if os.path.exists(image_path):
        os.remove(image_path)
        # print("File has been deleted")
    # else:
        # print("File does not exist:", image_path)

