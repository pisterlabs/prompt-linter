import os
import re
import json
import boto3
from botocore.config import Config
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.agents import Tool, AgentExecutor, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document
from typing import Any, Dict, List, Optional,Union
from langchain.utilities import SerpAPIWrapper
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.memory import ConversationBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType



os.environ["SERPAPI_API_KEY"]="e94267b343a2985d25d7a9a65e1b31a6629a4b4860872c927b33c88674fa89d2"
#role based initial client#######
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"  # E.g. "us-east-1"
os.environ["AWS_PROFILE"] = "default"
#os.environ["BEDROCK_ASSUME_ROLE"] = "arn:aws:iam::687912291502:role/service-role/AmazonSageMaker-ExecutionRole-20211013T113123"  # E.g. "arn:aws:..."


parameters_bedrock = {
    "max_tokens_to_sample": 2048,
    #"temperature": 0.5,
    "temperature": 0,
    #"top_k": 250,
    #"top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
  
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]
        

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    client_kwargs["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID","")
    client_kwargs["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY","")
    
    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client



## for aksk bedrock
def get_bedrock_aksk(secret_name='chatbot_bedrock', region_name = "us-west-2"):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret['BEDROCK_ACCESS_KEY'],secret['BEDROCK_SECRET_KEY']

class BedrockModelWrapper(Bedrock):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt = "\nHuman: \n" + prompt + "\nAssistant:"   ## Satisfy Bedrock-Claude prompt requirements
        return super()._call(prompt, stop, run_manager, **kwargs)


class FullContentRetriever(BaseRetriever):
    doc_path={}
    def _get_content_type(self,doc_path:str):
        fullname=""
        for root, dirs, files in os.walk(doc_path):
            for f in files:
                fullname = os.path.join(root, f)
                ext = os.path.splitext(fullname)[1]
                self.doc_path[fullname]=ext
        print(self.doc_path)        

        
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        allDocs =[]
        for key,value in self.doc_path.items():
            #print("key:"+key+" value:"+value)
            if value == ".doc":
                word_loader = Docx2txtLoader(key)
                word_document = word_loader.load()
                allDocs.append(word_document)
            elif value == ".txt":
                txt_loader = TextLoader(key)
                txt_document = txt_loader.load()
                allDocs.append(txt_document)
            elif value == ".pdf":
                pdf_loader = PyPDFLoader(self.doc_path)
                pdf_document = pdf_loader.load()
                allDocs.append(pdf_document)
            else:
                pass
        return allDocs


##use customerized outputparse to fix claude not match 
##langchain's openai ReAct template don't have 
##final answer issue 
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        #print("cur step's llm_output ==="+llm_output)
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
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
                log=llm_output,
            )
            #raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)



       
#### initial the tools&llm&momory ...etc #############
retriever = FullContentRetriever()
ACCESS_KEY, SECRET_KEY=get_bedrock_aksk()
os.environ["AWS_ACCESS_KEY_ID"]=ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]=SECRET_KEY

retriever_tool = create_retriever_tool(
    retriever,
    "search enterprise documents",
    "useful for when you need to retreve documents regarding the user's question",
)
search = SerpAPIWrapper()
search_tool = Tool(
    name="search website",
    func=search.run,
    description="useful for when you need to answer questions by searching the website",
)
custom_tool_list = [retriever_tool,search_tool]

memory = ConversationBufferWindowMemory(k=2,memory_key="chat_history", input_key='input', output_key="output")
#新boto3 sdk只能session方式初始化bedrock
boto3_bedrock = get_bedrock_client(
    #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)
bedrock_llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs=parameters_bedrock)    
bedrock_llm_additional = BedrockModelWrapper(model_id="anthropic.claude-v2", 
                                          client=boto3_bedrock, 
                                          model_kwargs=parameters_bedrock)

output_parser = CustomOutputParser()


PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

customerized_instructions="""
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These are guidance on when to use a tool to solve a task, follow them strictly:
 - first use "search enterprise documents" tool to retreve the document to answer if need
 - then use "search website" tool to search the latest website to answer if need
"""

agent_executor = initialize_agent(custom_tool_list, bedrock_llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                                  verbose=True,max_iterations=5,
                                  handle_parsing_errors=True,
                                  memory=memory,
                                  agent_kwargs={
                                      "output_parser": output_parser,
                                      #'prefix':PREFIX,
                                      #'suffix':SUFFIX,
                                      'format_instructions':customerized_instructions
                                           }
                                 )