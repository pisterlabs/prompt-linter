from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from retriever import get_highest_similarity_score
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import json
import os
import openai
import dotenv
from logger import setup_logger
import logging  # Added import for logging

# Set up the logger with the correct level
logger = setup_logger('nlu_logger', level=logging.INFO)

dotenv.load_dotenv()

logger.info("Initializing OpenAI with API key...")
logger.info("Loading data from prompt_examples.json...")

# Initialize OpenAI
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY
logger.info("OpenAI initialized with API key.")

# Load the data from the JSON file
with open("ai_chat/Data/Prompt_Eg/prompt_examples.json", "r") as file:
    MORTGAGE_INTENTS = json.load(file)
logger.info("Data loaded from prompt_examples.json.")

# Create a list to store examples in the desired format
examples_list = []

# Iterate over each intent in MORTGAGE_INTENTS
for intent, intent_data_list in MORTGAGE_INTENTS.items():
    for i, intent_data in enumerate(intent_data_list):
        example = intent_data['query']
        entities_list = intent_data['entities']

        example_str = f"- {intent} Example {i+1}: {example}"
        if entities_list:
            entities_str = ", ".join(entities_list)
            example_str += f" [Entities: {entities_str}]"
        
        examples_list.append(example_str)

# Create the updated prompt template with the examples in JSON format
examples_str = "\n".join(examples_list)
template = f"""
Given a mortgage-related user query, identify Primary Intents, Secondary Intents and format the response as follows:

- Start with "Primary Intent:" followed by the identified primary intent.
- If there is a secondary intent, list it as "Secondary Intent:" followed by the identified secondary intent.
- List entities as comma-separated values after "Entities:".
- End with "Question:" followed by the original user query.
Question: {{input}}
{{agent_scratchpad}}"""

def get_known_entities():
    all_entities = []
    for intent_data_list in MORTGAGE_INTENTS.values():
        for intent_data in intent_data_list:
            entities_list = intent_data.get('entities', [])
            all_entities.extend(entities_list)
    return set(all_entities)

def calculate_confidence(input_query, output):
    parsed_response = extract_from_response(output)
    primary_intent = parsed_response.get('primary_intent', '')
    secondary_intents = parsed_response.get('secondary_intent', [])
    extracted_entities = parsed_response.get('entities', [])
    
    # Intent Confidence
    primary_intent_confidence = 1.0 if primary_intent in MORTGAGE_INTENTS else 0.0
    
    # Checking each secondary intent against MORTGAGE_INTENTS
    secondary_intent_confidence = all(intent in MORTGAGE_INTENTS for intent in secondary_intents)
    
    # Semantic Confidence (assuming the get_highest_similarity_score function remains unchanged)
    similarity = get_highest_similarity_score(input_query)
    semantic_confidence = min(similarity, 1.0)
    
    # Entity Confidence
    known_entities = get_known_entities()
    entity_overlap = len(set(extracted_entities) & known_entities)
    entity_confidence = entity_overlap / len(extracted_entities) if extracted_entities else 1.0
    
    # Overall confidence
    confidence = 0.5 * primary_intent_confidence + 0.40 * semantic_confidence + 0.10 * entity_confidence
    logger.info(f"Calculated confidence for query '{input_query}': {confidence}.")
    return confidence


class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(template=template, input_variables=["input", "agent_scratchpad"], tools=[])

# Output Parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        lines = llm_output.split('\n')

        # Initial values
        primary_intent = ""
        secondary_intent = None
        entities = ""
        input_query = ""  # Initialize at the beginning

        for line in lines:
            if "Primary Intent:" in line:
                primary_intent = line.split("Primary Intent:")[1].strip()
            elif "Secondary Intent:" in line:
                secondary_intent = line.split("Secondary Intent:")[1].strip()
            elif "Entities:" in line:
                entities = line.split("Entities:")[1].strip()
            elif "Question:" in line:
                input_query = line.split("Question:")[1].strip()

        confidence = calculate_confidence(input_query, llm_output)  # Pass both input_query and llm_output
        logger.info(f"Primary Intent: {primary_intent}, Secondary Intent: {secondary_intent}, Entities: {entities}, Confidence: {confidence}.")

        return AgentFinish(
            return_values={
                "primary_intent": primary_intent,
                "secondary_intent": secondary_intent,
                "entities": entities,
                "output": llm_output.strip(), 
                "confidence": confidence
            },
            log={"llm_output": llm_output}
        )

output_parser = CustomOutputParser()

# Set up LLM
llm = ChatOpenAI(model_name="ft:gpt-3.5-turbo-0613:personal::7sczKPwS", temperature=0.9)

# LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Set up the Agent
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservations"]
)

# Agent Executor (turning off verbose mode)
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=False)

def extract_from_response(output: str):
    """
    Extracts primary intent, secondary intent, and entities from the raw LLM output.
    """
    lines = output.strip().split("\n")
    extracted_data = {
        "primary_intent": "",
        "secondary_intent": [],
        "entities": []
    }

    for line in lines:
        if "Primary Intent:" in line:
            extracted_data["primary_intent"] = line.split("Primary Intent:")[1].strip()
        elif "Secondary Intent:" in line:
            # Splitting by comma to extract multiple secondary intents, if present
            secondary_intents = line.split("Secondary Intent:")[1].strip().split(",")
            extracted_data["secondary_intent"] = [intent.strip() for intent in secondary_intents]
        elif "Entities:" in line:
            entities = line.split("Entities:")[1].strip().split(",")
            extracted_data["entities"] = [entity.strip() for entity in entities]
    logger.info(f"Extracted primary intent: {extracted_data['primary_intent']}, secondary intent: {extracted_data['secondary_intent']}, entities: {extracted_data['entities']}.")
    return extracted_data

def test_agent():
    sample_queries = [
    "How do interest rates influence my monthly mortgage payments?",
    "What documents are required for refinancing my home?",
    "Can you explain the difference between a fixed-rate and an adjustable-rate mortgage?",
    "What's the process to apply for a jumbo loan, and what are the eligibility criteria?",
    "Are there any special programs for first-time homebuyers?"
]

    for query in sample_queries:
        raw_response = agent_executor.run({"input": query, "agent_scratchpad": ""})
        
        print(f"Raw Response: {raw_response}\n")
        
        parsed_response = extract_from_response(raw_response)
        primary_intent = parsed_response.get('primary_intent', 'Error')
        secondary_intent = parsed_response.get('secondary_intent', '')
        entities = parsed_response.get('entities', [])
        
        # Note the change here: providing both query and raw_response
        confidence = calculate_confidence(query, raw_response) 
        
        print(f"Query: {query}")
        print(f"Primary Intent Identified: {primary_intent}")
        if secondary_intent:
            print(f"Secondary Intent Identified: {secondary_intent}")
        print(f"Entities: {', '.join(entities)}")
        print(f"Confidence: {confidence}\n")

if __name__ == "__main__":
    test_agent()

def get_known_entities():
    all_entities = []
    for intent_data_list in MORTGAGE_INTENTS.values():
        for intent_data in intent_data_list:
            entities_list = intent_data.get('entities', [])
            all_entities.extend(entities_list)
    return set(all_entities)
