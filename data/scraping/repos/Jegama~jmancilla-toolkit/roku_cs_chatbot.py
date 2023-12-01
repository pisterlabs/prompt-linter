from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain import OpenAI, LLMChain
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import List, Union
# from elevenlabs import generate, play
from llama_index import (
    LLMPredictor,
    ServiceContext,
    GPTVectorStoreIndex
)
from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext

from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
import pandas as pd
import re

from dotenv import load_dotenv
load_dotenv()

# Azure OpenAI service Context
# llm = LLMPredictor(llm=AzureOpenAI(temperature=1, deployment_name='gpt-35-turbo', model_name='gpt-35-turbo'))
# embeddings = LangchainEmbedding(OpenAIEmbeddings(deployment="text-embedding-ada-002"))
# service_context = ServiceContext.from_defaults(llm_predictor=llm, chunk_size_limit=1024, embed_model=embeddings)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024)

cs_index = GPTVectorStoreIndex([], service_context=service_context)

docid_to_url = pd.read_json('cs_docid_to_url.json', typ='series').to_dict()

# rebuild storage context
print("Loading index from storage...")

# Main CS Index
cs_storage_context = StorageContext.from_defaults(persist_dir="cs_index")
cs_index = load_index_from_storage(cs_storage_context)
cs_query_engine = cs_index.as_query_engine()

# Error Codes Index
error_storage_context = StorageContext.from_defaults(persist_dir="error_codes_index")
error_codes_index = load_index_from_storage(error_storage_context)
error_codes_query_engine = error_codes_index.as_query_engine()

class SourceFormatter:
    def formater(self, response, source_nodes):
        """Get formatted sources text."""
        texts = []
        texts.append(response)
        for source_node in source_nodes[:3]:
            title = source_node.node.text.split('\n')[0]
            doc_id = source_node.node.doc_id or "None"
            try:
                # TODO add score
                source_text = f"\nSource:\nURL: <a href=\"{docid_to_url[doc_id]}\">{title}</a>"
            except:
                source_text = f"\nSource:\nFirst line: {title} \nDocID: {doc_id}"
            texts.append(source_text)
        return "\n".join(texts)
    
    def query_cs(self, question):
        response = cs_query_engine.query(question)
        return self.formater(response.response, response.source_nodes)
    
    def query_error_codes(self, question):
        response = error_codes_query_engine.query(question)
        return self.formater(response.response, response.source_nodes)
    
    def connect_to_human(self, question):
        return "Please connect with a human agent by going to https://support.roku.com/contactus.\nThank you for using Roku Support. Have a nice day!\n\nAnother alternative is connect with my human by email at jmancilla@roku.com or scheduling a call <a href=\"https://calendly.com/jgmancilla/phonecall\">here</a>."
    
    def audio_guide(self, question):
        return "When the screen reader shortcut is enabled, you can quickly press Star * button on Roku remote four times to turn the screen reader on or off from any screen.\n\nSource <a href=\"https://support.roku.com/article/231584647\">How to enable the text-to-speech screen reader on your Roku® streaming device</a>."

# If the first search doesn't work, try different keywords; for example, if the user wants to change the pin, you can search for "update pin" or "reset pin." 

template = """You are a friendly Roku customer support agent. People who talk with you might not be tech-savvy; you can break down the instructions into smaller, more manageable steps. For example, instead of providing a long list of actions to take, you could break down each step and explain it thoroughly in short, simple sentences. The "CS Vector Index" articles might have solutions for different devices.

Remember, not all problems can be solved on the device; if the article mentions "my.roku.com," they need to go to that website to change something on their account. Convert all the URLs on your response in the following format `<a href="https://support.roku.com/">Roku Support Site</a>.` 

Here are 3 scenarios where you must skip the action and give a "Final Answer." Apply them in the following order:
If a user thanks you or shows kindness, skip action and respond accordingly.
If the question is unrelated to Roku, please reply politely, saying that you can only answer questions related to Roku. 
If the question is only 1 or 2 words, please reply politely, saying you need more information to help them.

You have access to the following tools: {tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times, if you don't find the answer by then, politely tell the user to connect with a human agent)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. Include the source from the Observation.

Begin! Remember to be friendly and explain things thoroughly with simple language. Always make sure the source URL is correct. If you use "search," return the URL from the website where you took the answer, and tell the user you could not find the answer on our official documentation. Let's think step by step to ensure we have the correct answer.

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        # TODO add criticism to the agent
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))

formatter = SourceFormatter()
cs_index_config = Tool(
    func=formatter.query_cs, 
    name=f"CS Vector Index",
    description=f"Your primary tool, useful for when you want to answer queries using the official documentation on the Roku Customer Support site"
)

error_code_index_config = Tool(
    func=formatter.query_error_codes, 
    name=f"Error Codes Index",
    description=f"useful for when you want to answer queries about error codes"
)

human_config = Tool(
    func=formatter.connect_to_human,
    name=f"Connect to Human",
    description=f"useful for when you want to connect the user to a human agent"
)

audio_guide_config = Tool(
    func=formatter.audio_guide,
    name=f"Audio Guide",
    description=f"useful for when you want to answer queries about the audio guide, the screen reader, or when the participant says that the TV is talking to them"
)

search = SerpAPIWrapper()
search_config = Tool(
        name = "search",
        func=search.run,
        description="This is your last resort, useful when you can not find the answer on the official Roku Customer Support site. You should ask targeted questions"
    )

tools = [cs_index_config, error_code_index_config, audio_guide_config, human_config, search_config]

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

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

output_parser = CustomOutputParser()

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(
    llm=OpenAI(temperature=1), 
    prompt=prompt,
)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


while True:
    text_input = input("User: ")
    response = agent_executor.run(input=text_input)
    print(f'Agent: {response}')
