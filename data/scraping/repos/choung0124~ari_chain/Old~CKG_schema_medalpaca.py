from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, initialize_agent, AgentType, load_tools
from langchain import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging
from langchain.llms import HuggingFacePipeline, TextGen
from langchain.memory import SimpleMemory, VectorStoreRetrieverMemory, ConversationKGMemory, CombinedMemory, ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from Custom_Agent import get_umls_id, CustomLLMChain, PubmedAgentOuputParser, create_ggml_model, create_gptq_pipeline, PubmedAgentPromptTemplate, FinalAgentOutputParser
import re
from customGraphCypherQA import KnowledgeGraphRetrieval
from prompts import Entity_Extraction_Template, PUBMED_AGENT_TEMPLATE_VICUNA, Med_alpaca_QA_PROMPT_TEMPLATE, PUBMED_AGENT_TEMPLATE, Vicuna_QA_PROMPT_TEMPLATE, Entity_Extraction_Template_alpaca
from custom_neo4j_class_langchain import Neo4jGraph
from langchain.tools import PubmedQueryRun, DuckDuckGoSearchRun
import torch
import sys
import gc
import transformers
logging.set_verbosity(logging.CRITICAL)
#model_name_or_path = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
#model_basename = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit.act.order"
#model_name_or_path = "TheBloke/medalpaca-13B-GPTQ-4bit"
#model_basename = "medalpaca-13B-GPTQ-4bit-128g.latest.act-order"
#tokenizer = AutoTokenizer.from_pretrained("oobabooga/llama-tokenizer")
#pipe = create_gptq_pipeline(model_name_or_path, model_basename, tokenizer)
#llm = HuggingFacePipeline(pipeline=pipe)

model_url = "https://server-how-powers-democracy.trycloudflare.com"
llm = TextGen(model_url=model_url)
Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template_alpaca, input_variables=["input"])
Entity_extraction_chain = CustomLLMChain(prompt=Entity_extraction_prompt, llm=llm, output_key="output",)
result = Entity_extraction_chain.run("Could there be a negative drug-drug interaction between mirodenafil and donepezil?")
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

del llm
del Entity_extraction_chain
torch.cuda.empty_cache()
gc.collect()  # Add this line to force garbage collection

model_name_or_path2 = "TheBloke/Nous-Hermes-13B-GPTQ"
model_basename2 = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
#model_name_or_path2 = "TheBloke/minotaur-15B-GPTQ" 
#model_basename2 = "gptq_model-4bit-128g"
#model_name_or_path2 = "TheBloke/mpt-30B-instruct-GGML" 
#model_basename2 = "mpt-30b-instruct.ggmlv0.q8_0.bin"
tokenizer2 = AutoTokenizer.from_pretrained(model_name_or_path2)

#pipe2 = create_gptq_pipeline(model_name_or_path2, model_basename2, tokenizer2)
#llm2 = HuggingFacePipeline(pipeline=pipe2)
llm2 = create_gptq_pipeline(model_name_or_path2, model_basename2, tokenizer2)

uri = "neo4j://localhost:7687"
username = "neo4j"
password = "NeO4J"
#driver = GraphDatabase.driver(url, auth=(username, password))

graph_clustering = KnowledgeGraphRetrieval(uri, username, password)
result = graph_clustering._call(names_list)
print(list(result))
print(result["result"])
context = result["result"]
newnew = []
for record in context:
    source_name = record["sourceName"]
    target_name = record["targetName"]
    paths = record["pathElements"]

    # Generate a list of formatted paths
    formatted_paths = [" -> ".join(path_element) for path in paths for path_element in path]

    # Combine the paths into a single string using commas
    paths_string = ", ".join(formatted_paths)

    # Create a sentence with the source, target, and paths
    sentence = f"The relationship between {source_name} and {target_name} involves the following paths: {paths_string}."
    newnew.append(sentence)

# Print the output
for sentence in newnew:
    print(sentence)

search = PubmedQueryRun()
duck_search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name = "Retrieve articles from Pubmed",
        func=search.run,
        description="useful for when you need to find information from Pubmed"
    ),
    Tool(
        name = "DuckDuckGo Searching",
        func=duck_search.run,
        description="useful for when you need to find more general information"
    )
]

prompt = PromptTemplate(
    template=Med_alpaca_QA_PROMPT_TEMPLATE,
    input_variables=["input", "context"]
)
output_parser = FinalAgentOutputParser()
llm_chain = LLMChain(llm=llm2, prompt=prompt)

new_context = llm_chain.predict(input="""what is the relationship between alzheimer's and autophagy?""",
                       context=long_string)
print(new_context)
del llm_chain
#del pipe2
del llm2
torch.cuda.empty_cache()
gc.collect()

prompt2 = PubmedAgentPromptTemplate(
    template=PUBMED_AGENT_TEMPLATE_VICUNA,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

memory = ConversationBufferMemory()
model_name_or_path3 = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
model_basename3 = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit.act.order"
tokenizer3 = AutoTokenizer.from_pretrained(model_name_or_path3)
pipe3 = create_gptq_pipeline(model_name_or_path3, model_basename3, tokenizer3)
llm3 = HuggingFacePipeline(pipeline=pipe3)

Pubmed_agent_chain = LLMChain(llm=llm3, prompt=prompt2)
output_parser = PubmedAgentOuputParser()

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=Pubmed_agent_chain,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
    output_parser=output_parser
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
agent_executor.run(input=new_context)


