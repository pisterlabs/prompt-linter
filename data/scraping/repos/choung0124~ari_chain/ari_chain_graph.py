from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, AgentType, initialize_agent
from langchain.chains import TransformChain, SimpleSequentialChain
from langchain.graphs import Neo4jGraph
from langchain import LLMChain
from typing import Any, List, Dict, Tuple, Union
from langchain.prompts import BaseChatPromptTemplate, StringPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.memory import SimpleMemory, VectorStoreRetrieverMemory, ConversationKGMemory, CombinedMemory
from neo4j import GraphDatabase
from langchain.prompts.prompt import PromptTemplate
from auto_gptq import AutoGPTQForCausalLM
from UMLS_agent import get_umls_id, parse_llm_output, Entity_Extraction_Template, CustomLLMChain
import re
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from customGraphCypherQA import CustomGraphCypherQAChain, CYPHER_GENERATION_TEMPLATE, CYPHER_GENERATION_TEMPLATE_2
from prompts import CYPHER_GENERATION_TEMPLATE, CYPHER_GENERATION_TEMPLATE_2
#model_name_or_path = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
#model_basename = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit.act.order"
model_name_or_path = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
model_basename = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit.act.order"
use_triton=False

url = "neo4j://localhost:7687"
username = "neo4j"
password = "NeO4J"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
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
    max_length=1024,
    temperature=0.2,
    top_p=0.95,
    repetition_penalty=1.15
    )

llm = HuggingFacePipeline(pipeline=pipe)
Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template, input_variables=["input"])
Entity_extraction_chain = CustomLLMChain(prompt=Entity_extraction_prompt, llm=llm, output_key="output",)


CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question", "UMLS_context"], template=CYPHER_GENERATION_TEMPLATE
)
CYPHER_GENERATION_PROMPT_2 = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE_2
)

with GraphDatabase.driver(url, auth=(username, password)) as driver:
    result = Entity_extraction_chain.run("""What is the relationship between alzheimer's and autophagy?""")
    print("Raw result:", result)
    entities = result
    entities_umls_ids = {}

    for entity in entities:
        umls_id = get_umls_id(entity)
        entities_umls_ids[entity] = umls_id

    output_string = ""
    for entity, umls_info_list in entities_umls_ids.items():
        if umls_info_list:
            umls_info = umls_info_list[0]
            match = re.search(r"Name: (.*?) UMLS_CUI: (\w+)", umls_info)
            if match:
                umls_name = match.group(1)
                umls_cui = match.group(2)
                output_string += f"Name: {umls_name} UMLS_CUI: {umls_cui}, "
            else:
                output_string += f"Name: {entity} UMLS_CUI: Not found, "
        else:
            output_string += f"Name: {entity} UMLS_CUI: Not found, "
    # Remove the trailing comma and space
    output_string = output_string.rstrip(', ')

    print(output_string)
    UMLS_context = output_string

    graph = Neo4jGraph(driver=driver)
    graph.refresh_schema()
    print(graph.get_schema)
    model_name_or_path = "TheBloke/minotaur-15B-GPTQ"
    model_basename = "gptq_model-4bit-128g"
    use_triton=False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
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
        max_length=8192,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15
        )
    llm = HuggingFacePipeline(pipeline=pipe)
    graph_chain = CustomGraphCypherQAChain.from_llm(llm, graph=graph, verbose=True, cypher_prompt=CYPHER_GENERATION_PROMPT, return_intermediate_steps=True)
    inputs = {
        "question": "What is the relationship between alzheimer's and diabetes?",
        "UMLS_context": UMLS_context,
    }
    result = graph_chain(inputs)
    print(f"Intermediate steps: {result['intermediate_steps']}")
    print(f"Final answer: {result['result']}")



