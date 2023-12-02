from flask import Flask
from flask_socketio import SocketIO, emit
import time
import json
from datetime import datetime, timedelta
import pytz
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
import os
from getpass import getpass
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import re
import requests
import ast
import subprocess
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from substrateinterface import SubstrateInterface, Keypair
from substrateinterface.contracts import ContractCode, ContractInstance
from substrateinterface.exceptions import SubstrateRequestException
from tools.custom_pola_tools import (GetAccountBalanceTool,
                                        SendSubstrateBalanceTool,
                                        ListAllTransactionsTool,
                                        GetTransferDetailsTool,
                                        GetERC20TotalSupplyTool,
                                        GetERC20OfUserTool,
                                        TransferERC20ToUserTool)
from dotenv import load_dotenv

class GenerateInkPolkadotContractInput(BaseModel):
    """Inputs for generate_ink_polkadot_contract"""

    contract_description: str = Field(
        description=
        "A description in simple english of what you would like the contract to do"
    )

def send_data_to_user(prompt):
    data = data_global
    payload = dict()
    payload['prompt'] = prompt
    session = {
        'user': {
            'name': 'Sybil AI',
            'email': data['session']['user']['email'],
            'image': 'https://i.imgur.com/usI3OTw.png'
        }
    }
    payload['session'] = session
    payload['chatId'] = data['chatId']
    now_utc = datetime.now(pytz.timezone('UTC')) + timedelta(seconds=2)
    payload['createdAt'] = now_utc.isoformat()

    emit('response', payload)

def getPolkaDocs(query):
    '''
    Get the polkadot docs for a query
    '''
    global qa_docs
    result = qa({"question": query, "chat_history": []})

    return result["answer"]


class GenerateInkPolkadotDocsInfoInput(BaseModel):
    """Inputs for generate_ink_polkadot_docs_info"""

    query: str = Field(
        description=
        "A description in simple english of what information you would like aobut polkadot or substrate"
    )

class GenerateInkPolkadotDocsInfoTool(BaseTool):
    name = "generate_ink_polkadot_docs_info"
    description = """
        Useful when you want to get information pertaining to polkadot or substrate.
        """
    args_schema: Type[BaseModel] = GenerateInkPolkadotDocsInfoInput

    def _run(self, query: str):
        result = getPolkaDocs(query)
        return result

    def _arun(self, account_address: str):
        raise NotImplementedError(
            "get_current_stock_price does not support async")


class GenerateInkPolkadotContractTool(BaseTool):
    name = "generate_ink_polkadot_contract"
    description = """
        Useful when you want to generate a polkadot contract in ink or just an ink contract.
        The contract description is a description of what you would like the contract to do.
        
        This also deploys the code to Shibuya Testnet.

        returns the contract address
        """
    args_schema: Type[BaseModel] = GenerateInkPolkadotContractInput

    def _run(self, contract_description: str):
        address = genCompileDeployContract(contract_description)
        return address

    def _arun(self, account_address: str):
        raise NotImplementedError(
            "get_current_stock_price does not support async")
    
def run_command(command):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               shell=True)
    output, error = process.communicate()

    if process.returncode != 0:
        print(f"Error occurred: {error.decode().strip()}")
    else:
        print(f"Output: {output.decode().strip()}")

def genCompileDeployContract(description: str):
    '''
    Generate, compile and deploy a contract to Shibuya Testnet
    '''
    load_dotenv()
    mnemonic = os.getenv("MNEMONIC")
    questions = [
        f"""
    Give me a basic ink contract code, 
    Description of code = {description}
    """ + """
    -- There should be no print statments in the contract return everything.
    -- enclose the contract content in #startContract# and #endContract#
    -- follow this basic format when generating the code

    #startContract#
    #[ink::contract]
    mod contract_name \{

    }
    #endContract#
    """
    ]

    chat_history = []
    global qa

    send_data_to_user("starting code generation")
    # extrating the contract code from the result
    result = qa({"question": questions[0], "chat_history": chat_history})
    chat_history.append((questions[0], result["answer"]))

    pattern = r'#startContract#(.*?)#endContract#'
    pattern2 = r'```rust(.*?)```'
    contract_code = re.search(pattern, result["answer"], re.DOTALL)
    res = ""

    if contract_code:
        res = contract_code.group(1).strip()
    else:
        contract_code = re.search(pattern2, result["answer"], re.DOTALL)
        if contract_code:
            res = contract_code.group(1).strip()
    res = r"""#![cfg_attr(not(feature = "std"), no_std, no_main)]""" + '\n' + res
    post_process_code = re.sub(r'^\s*use.*\n?', '', res, flags=re.MULTILINE)

    post_process_code = re.sub(r'^\s*struct',
                               'pub struct',
                               post_process_code,
                               flags=re.MULTILINE)

    post_process_code = re.sub(r'^\s*#\s*\[derive\(.*\n?',
                               '',
                               post_process_code,
                               flags=re.MULTILINE)

    print(post_process_code)

    # generating the constructor args
    new_function_pattern = r'(pub fn new.*?\))'
    new_function_match = re.search(new_function_pattern, post_process_code,
                                   re.DOTALL)

    if new_function_match:
        res = new_function_match.group(0)

    send_data_to_user(post_process_code)
    send_data_to_user("Compiling code")


    print(res)
    chat = ChatOpenAI()
    messages = [
        SystemMessage(content=r"""
        give the argumentsvalues for "pub fn new(value1: i32, value2: i32)" in the form of a dictionary
        -- Just the dictionary
        -- No need fore explanation or additional code
        -- empty dictionary is also fine
        -- for invalid input empty dictionary will be returned
        example: 
        Input: pub fn new(coolVal: i32)
        Output: {"coolVal": 1}"""),
        HumanMessage(content=f"{res}")
    ]
    constructor_args = ast.literal_eval(chat(messages).content)

    with open('code/lib.rs', 'w') as file:
        file.write(post_process_code)

    # compiling contract
    print(run_command("cd code && cargo contract build"))

    # Upload WASM code
    code = ContractCode.create_from_contract_files(
        metadata_file=os.path.join(os.getcwd(), 'code/target/ink',
                                   'my_contract.json'),
        wasm_file=os.path.join(os.getcwd(), 'code/target/ink',
                               'my_contract.wasm'),
        substrate=substrate_relay)

    send_data_to_user(res + "\n" + "Deploying Code")
    # Deploy contract
    print('Deploy contract...')
    contract = code.deploy(keypair=Keypair.create_from_mnemonic(mnemonic),
                           constructor="new",
                           args=constructor_args,
                           value=0,
                           gas_limit={
                               'ref_time': 25990000000,
                               'proof_size': 1199000
                           },
                           upload_code=True)

    return contract.contract_address

load_dotenv()


embeddings = OpenAIEmbeddings()
db = DeepLake(
    dataset_path=f"hub://commanderastern/polka-code-3",
    read_only=True,
    embedding_function=embeddings,
)

retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 20
retriever.search_kwargs["maximal_marginal_relevance"] = True
retriever.search_kwargs["k"] = 20

# Polka docs retreival
db_docs = DeepLake(
    dataset_path=f"hub://commanderastern/polka-docs",
    read_only=True,
    embedding_function=embeddings,
)

retriever_docs = db_docs.as_retriever()
retriever_docs.search_kwargs["distance_metric"] = "cos"
retriever_docs.search_kwargs["fetch_k"] = 20
retriever_docs.search_kwargs["maximal_marginal_relevance"] = True
retriever_docs.search_kwargs["k"] = 20

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

substrate_relay = SubstrateInterface(url="wss://shibuya-rpc.dwellir.com")
base_url = "https://shibuya.api.subscan.io"
qa = ""
qa_docs = ""

data_global = ""

@socketio.on('query')
def handle_query(data):

    print(data)
    weather_data = {"type": "weather", "content": "It's sunny today!"}
    emit('response', weather_data)
    time.sleep(10)

    news_data = {"type": "news", "content": "Latest news update here!"}
    emit('response', news_data)
    time.sleep(10)

    other_data = {"type": "other", "content": "Other type of data!"}
    emit('response', other_data)


@socketio.on('print')
def handle_query(data):

    # check if data is a json
    try:
        data = json.loads(data)
    except:
        return
    print(data)

    global data_global
    data_global = data

    # check if it open ai key is not null
    if data['openAIKey'] is None or data['openAIKey'] == "":
        print("no open ai key")
        return
    if data['mnenonic'] is None or data['mnenonic'] == "":
        print("no mnenonic")
        return
    # reinitialize agent everytime a query is made
    os.environ["OPENAI_API_KEY"] = data['openAIKey']
    os.environ["MNEMONIC"] = data['mnenonic']
    llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
    tools = [
        GetAccountBalanceTool(),
        GenerateInkPolkadotContractTool(),
        SendSubstrateBalanceTool(),
        ListAllTransactionsTool(),
        GetTransferDetailsTool(),
        GetERC20TotalSupplyTool(),
        GetERC20OfUserTool(),
        TransferERC20ToUserTool(),
        GenerateInkPolkadotDocsInfoTool()
    ]
    agent = initialize_agent(tools,
                             llm,
                             agent=AgentType.OPENAI_FUNCTIONS,
                             verbose=True)

    model = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k")  # 'ada' 'gpt-3.5-turbo' 'gpt-4',
    global qa
    global qa_docs
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
    qa_docs = ConversationalRetrievalChain.from_llm(model, retriever=retriever_docs)

    payload = dict()
    # payload['prompt'] = data['prompt'] + "nisoo"
    mnemonic = data['mnenonic']
    openai = data['openAIKey']
    session = {
        'user': {
            'name': 'Sybil AI',
            'email': data['session']['user']['email'],
            'image': 'https://i.imgur.com/usI3OTw.png'
        }
    }
    payload['session'] = session
    payload['chatId'] = data['chatId']

    payload['prompt'] = agent.run(data['prompt'])

    now_utc = datetime.now(pytz.timezone('UTC')) + timedelta(seconds=2)
    payload['createdAt'] = now_utc.isoformat()
    emit('response', payload)


if __name__ == '__main__':
    socketio.run(app)
