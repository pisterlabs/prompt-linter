from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, AgentType
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from utils import extract_from_response
from specialist_agents import eligibility_agent, loan_comparison_agent, amortization_agent, scenario_agent
from langchain.chat_models import ChatOpenAI
from confidence_calculator import ConfidenceCalculator
from GeneralInfo_agent import GeneralInfoAgent  # Import GeneralInfoAgent
from database_prep import DatabaseManager  # Import DatabaseManager
from typing import List, Union, Optional
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import os
import dotenv
import openai
from logger import setup_logger
import logging  # Added import for logging

# Set up the logger with the correct level
logger = setup_logger('nlu_logger', level=logging.INFO)  # Reduced the logging level to INFO

dotenv.load_dotenv()

logger.info("Initializing OpenAI with API key...")

# Initialize OpenAI
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY
logger.info("OpenAI initialized with API key.")
API_KEY = os.getenv("AIRTABLE_API_KEY")
BASE_ID = os.getenv("AIRTABLE_BASE_ID")
TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

template = f"""
Given a mortgage-related user query, identify Primary Intents, Secondary Intents, and Specialist Agent as per the training provided and format the response as follows:

- Start with "Primary Intent:" followed by the identified primary intent.
- Identify secondary intent, list it as "Secondary Intent:" followed by the identified secondary intent.
- List entities as comma-separated values after "Entities:".
- End with "Specialist Agent:" followed by the identified specialist agent best suited to handle the query.

Question: {{input}}
{{agent_scratchpad}}"""

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

# Initialize Database Manager (Assuming DatabaseManager class exists)
db_manager = DatabaseManager(BASE_ID, API_KEY, TABLE_NAME)  # Initialize DatabaseManager

# Initialize the GeneralInfoAgent class
general_info_agent = GeneralInfoAgent()

# Define agents (Assuming these agents are already initialized)
agents = {
    'Eligibility': eligibility_agent,
    'GeneralInfo': general_info_agent,  # Add this line
    'LoanComparison': loan_comparison_agent,
    'Amortization': amortization_agent,
    'Scenario': scenario_agent
}

# Initialize ConfidenceCalculator with known agents
confidence_calculator = ConfidenceCalculator(known_agents=agents.keys())

# RouterAgent Class
class RouterAgent:
    def route(self, user_query):
        response = None  # Initialize response to None or some default value
        confidence = 0.0  # Initialize confidence to a default value
        specialist_agent = None  # Initialize specialist_agent to None or some default value
        try:
            output_parser.set_user_query(user_query)
            raw_response = agent_executor.run({"input": user_query, "agent_scratchpad": ""})
            parsed_response = output_parser.parse(raw_response)
            primary_intent = parsed_response.return_values.get('primary_intent', 'Error')
            secondary_intent = parsed_response.return_values.get('secondary_intent', '')
            entities = parsed_response.return_values.get('entities', [])
            specialist_agent = parsed_response.return_values.get('specialist_agent', '')

            # Moved the confidence calculation to here, to avoid multiple retriever calls
            confidence = confidence_calculator.calculate_confidence(user_query, raw_response, specialist_agent, log=True)
            
            selected_agent = agents.get(specialist_agent, general_info_agent)
            response = selected_agent.func(user_query, entities)

            # Inserting record into DB
            db_manager.insert_interaction(
                user_query=user_query,
                mira_response=response,
                primary_intents=primary_intent,
                secondary_intents=','.join(secondary_intent) if isinstance(secondary_intent, list) else secondary_intent,
                entities=entities,
                action_taken=specialist_agent,
                confidence_score=confidence,
                session_id="N/A"  # Placeholder
            )

        except Exception as e:
            logger.error(f"Error occurred while routing query '{user_query}': {e}")
        return response, confidence, specialist_agent
    
# Output Parser
class CustomOutputParser(AgentOutputParser):
    user_query: Optional[str] = None  # Specify type hint and set default to None

    def set_user_query(self, user_query):
        self.user_query = user_query  # Set user_query as an instance variable

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        lines = llm_output.split('\n')
        parsed_data = extract_from_response(llm_output)
        primary_intent = parsed_data['primary_intent']
        secondary_intent = parsed_data['secondary_intent']
        entities = parsed_data['entities']
        specialist_agent = parsed_data['specialist_agent']

        for line in lines:
            if "Primary Intent:" in line:
                primary_intent = line.split("Primary Intent:")[1].strip()
            elif "Secondary Intent:" in line:
                secondary_intent = line.split("Secondary Intent:")[1].strip()
            elif "Entities:" in line:
                entities = line.split("Entities:")[1].strip().split(", ")
            elif "Specialist Agent:" in line:
                specialist_agent = line.split("Specialist Agent:")[1].strip()

        user_query = self.user_query.strip()
        # Calculate confidence
        confidence = confidence_calculator.calculate_confidence(user_query, llm_output, specialist_agent, log=False)  # use user_query
        logger.info(f"Primary Intent: {primary_intent}, Secondary Intent: {secondary_intent}, Entities: {entities}, Specialist Agent: {specialist_agent}, Confidence: {confidence}.")

        return AgentFinish(
            return_values={
                "primary_intent": primary_intent,
                "secondary_intent": secondary_intent,
                "entities": entities,
                "specialist_agent": specialist_agent,
                "output": llm_output.strip(),
                "confidence": confidence
            },
            log= llm_output
        )

output_parser = CustomOutputParser()

# Set up LLM
llm = ChatOpenAI(model_name="ft:gpt-3.5-turbo-0613:personal::7scg7esv", temperature=0.7)

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

# Initialize RouterAgent
router_agent = RouterAgent()

def test_agent():
    sample_queries = [
        "Can you provide some information about FHA loans?",
        "What are the documents required to start the pre-approval process?",
        "How to check if I am eligible for a mortgage",
        "What is the status of my Loan Application",
        "How do I submit my documents?",
        "I have scenario I want to discuss?"
    ]

    for query in sample_queries:
        response, confidence, specialist_agent = router_agent.route(query)  # This will internally handle everything

        # Print results
        print(f"Query: {query}")
        print(f"Response: {response}")
        print(f"Specialist Agent: {specialist_agent}")
        print(f"Confidence: {confidence}")

if __name__ == "__main__":
    test_agent()