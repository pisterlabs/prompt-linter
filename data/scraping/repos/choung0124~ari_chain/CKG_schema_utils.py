from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, AgentType, initialize_agent
from langchain.chains import TransformChain, SimpleSequentialChain
from langchain import LLMChain
from typing import Any, List, Dict, Tuple, Union
from langchain.prompts import BaseChatPromptTemplate, StringPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.memory import SimpleMemory, VectorStoreRetrieverMemory, ConversationKGMemory, CombinedMemory
from neo4j import GraphDatabase
from langchain.prompts.prompt import PromptTemplate
from auto_gptq import AutoGPTQForCausalLM
from Custom_Agent import get_umls_id, CustomLLMChain, FinalAgentOutputParser, FinalAgentPromptTemplate
import re
from customGraphCypherQA import KnowledgeGraphRetrieval
from prompts import Entity_Extraction_Template, QA_PROMPT_TEMPLATE
from custom_neo4j_class_langchain import Neo4jGraph
from langchain.tools import PubmedQueryRun
#from langchain.graphs import Neo4jGraph
model_name_or_path = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
model_basename = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit.act.order"
use_triton=False
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

pipe = pipeline(
   "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    temperature=0.2,
    top_p=0.95,
    repetition_penalty=1.15
    )

llm = HuggingFacePipeline(pipeline=pipe)
Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template, input_variables=["input"])
Entity_extraction_chain = CustomLLMChain(prompt=Entity_extraction_prompt, llm=llm, output_key="output",)
result = Entity_extraction_chain.run("""What is the relationship between alzheimer's and autophagy?""")
print("Raw result:", result)
entities = result
entities_umls_ids = {}

for entity in entities:
    umls_id = get_umls_id(entity)
    entities_umls_ids[entity] = umls_id

names_list = []

for entity, umls_info_list in entities_umls_ids.items():
    if umls_info_list:
        umls_info = umls_info_list[0]
        match = re.search(r"Name: (.*?) UMLS_CUI: (\w+)", umls_info)
        if match:
            umls_name = match.group(1)
            umls_cui = match.group(2)
            names_list.append(umls_name)
        else:
            names_list.append(entity)
    else:
        names_list.append(entity)
# Remove the trailing comma and space
print(names_list)

llm = HuggingFacePipeline(pipeline=pipe)

uri = "neo4j://localhost:7687"
username = "neo4j"
password = "NeO4J"
#driver = GraphDatabase.driver(url, auth=(username, password))
llm = HuggingFacePipeline(pipeline=pipe)

graph_clustering = KnowledgeGraphRetrieval(uri, username, password)
result = graph_clustering._call(names_list)
print(result["result"])
context = result["result"]
print(context)
search = PubmedQueryRun()
tools = [
    Tool(
        name = "Pubmed_Search",
        func=search.run,
        description="useful for when you need to find evidence to support your answer"
    )
]

prompt = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["input", "context"]
)
output_parser = FinalAgentOutputParser()
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
agent_executor.run(input="""what is the relationship between alzheimer's and autophagy?""",
                       context=context)


