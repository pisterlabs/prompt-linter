"""
File that contains the logic for risk analysis.
"""
import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.agents import Tool
from langchain import SerpAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain import LLMChain
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents import Tool, AgentExecutor
from langchain import SerpAPIWrapper, LLMChain
from langchain.agents import AgentType, initialize_agent
from langchain import LLMChain

from src.logic.langchain_tools.tool_process_thought import process_thoughts
from src.logic.langchain_tools.tool_get_risk_scoring_system import get_risk_scoring_system
from src.logic.langchain_tools.tool_query_risk_types import ToolSearchRiskTypes
from src.logic.helper_functionality.text_summarization import TextSummarizer

from src.logic.config import secrets as config_secrets

from db.database_connector import DatabaseConnector

class RiskAnalysis():
    """
    Class that contains the logic for risk analysis.
    """

    def __init__(self) -> None:
        self.text_summarizer: TextSummarizer = TextSummarizer()
        self.combined_result: str = ""

    def createDBConnection(self, db_name: str = "risk.db") -> DatabaseConnector:
        """
        Open a connection to the database and return a connection object.
        """
        db = DatabaseConnector(db_name)
        db.open()
        return db

    def analysis(self, company: str, news: str, message_type = None) -> str:
        """
        Perform risk analysis for a given company based on a potential focus and the provided content.

        :param company: The name of the company for which risk analysis is to be performed.
        :param news: Text content to be analyzed for potential risks.
        :return: A string containing the result of the analysis.
        :raise ValueError: If arg company is not a string or if the string is empty.
        :raise ValueError: If arg news is not a string or if the string is empty.
        """
        if not isinstance(company, str) or not company:
            raise ValueError("Argument company must be a non empty string")
        if not isinstance(news, str) or not news:
            raise ValueError("Argument news must be a non empty string")
        try:
            search: SerpAPIWrapper = SerpAPIWrapper(serpapi_api_key = config_secrets.read_serpapi_credentials())
            riskType: ToolSearchRiskTypes = ToolSearchRiskTypes()
            wikipedia: WikipediaAPIWrapper = WikipediaAPIWrapper()
            tools_risk_types = [
                Tool(
                    name = "Search",
                    func = search.run,
                    description = "useful for when you need to answer questions about current events. this tool should not be used to searhc for news articles",
                ),
                Tool(
                    name = "Thought Processing",
                    func = process_thoughts,
                    description = """useful for when you have a thought that you want to use in a task, 
                    but you want to make sure it's formatted correctly"""
                ),
                Tool(
                    name = "Get Risk Type",
                    func = riskType.run_find_type,
                    description = """useful when you need to find a specific risk type. the input should be an object of interest 
                    (e.g. price increases, brand reputation, decrease in user demand, etc.)"""
                ),
                Tool(
                    name = "Get Risk Type Description",
                    func = riskType.run,
                    description = """useful when you need detailed information for a specific risk type. the input should be 
                    the name of the risk type (e.g. market risk)"""
                ),
                Tool(
                    name = "Wikipedia",
                    func = wikipedia.run,
                    description = "useful for when you need to detailed information about a topic"
                )
            ]
            tools_risk_scoring_system = [
                Tool(
                    name = "Search",
                    func = search.run,
                    description = "useful for when you need to answer questions about current events",
                ),
                Tool(
                    name = "Thought Processing",
                    func = process_thoughts,
                    description = """useful for when you have a thought that you want to use in a task, 
                    but you want to make sure it's formatted correctly"""
                ),
                Tool(
                    name = "Get Risk Scoring System",
                    func = get_risk_scoring_system,
                    description = """useful when you need information about how to score risks (likelihood and impact)). takes a single input formatted as a question"""
                )
            ]
            system_template = """You are a helpful assistant. Your job is to read a news article and return its key point.
            News: {news}"""
            system_message_prompt: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template(system_template)
            if(message_type and message_type == "analysis"):
                db: DatabaseConnector = self.createDBConnection()
                query = "SELECT message FROM prompts WHERE type = ?"
                result = db.c.execute(query, (message_type,))
                row = result.fetchone()
                if row:
                    human_template = row[0]
                else:
                    human_template = "Please identify the key points of the news article."
            else:
                human_template = "Please identify the key point of the news article."
            human_message_prompt: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(human_template)
            chat_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt]
            )
            llm: ChatOpenAI = ChatOpenAI(
                model="gpt-4",
                temperature = 0, 
                client = chat_prompt, 
                openai_api_key = config_secrets.read_openai_credentials()
            )
            keypoint_chain: LLMChain = LLMChain(llm=llm, prompt=chat_prompt)
            keypoints: str = keypoint_chain.run(news=news)
            logging.info("Done reading article.")

            model_risk_score: ChatOpenAI = ChatOpenAI(model="gpt-4", temperature=0)
            planner_risk_score = load_chat_planner(model_risk_score)
            executor_risk_score = load_agent_executor(model_risk_score, tools_risk_types, verbose=True)
            agent_risk_score = PlanAndExecute(planner=planner_risk_score, executor=executor_risk_score, verbose=True)
            risk_analysis: str = agent_risk_score.run("""You are a helpful assistant. Please identify the risks for the 
            company {company} based on this statement: {keypoints}. Report each identified risk type (max. 3) and support your decision
            by providing explanations.""".format(company = company, keypoints = keypoints))
            logging.info("Done risk analysis.")
        
            system_template = "You are a helpful assistant. Your job is to award risk scores to identified risks."
            system_message_prompt: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template(system_template)
            example_message_human = SystemMessagePromptTemplate.from_template("""Please score the risks for the company 
            Apple as identified in this risk assessment: The potential risks for Apple include a decrease in demand 
            for the iPhone 15 range due to the higher prices, which could lead to lower sales and revenue. 
            Additionally, if international customers are facing their second successive major price increase, 
            this could lead to a negative impact on Apple's reputation and customer loyalty. Your final answer 
            should include a detailed explanation for your reasoning.""", additional_kwargs={"name": "example_user"})
            example_message_ai = SystemMessagePromptTemplate.from_template("""Based on the identified risks for Apple in the given 
            risk assessment, the risk of a decrease in demand for the iPhone 15 range due to the higher prices has a 
            likelihood rating of "Likely" and an impact rating of "Major". Therefore, the risk score for this risk is 
            4 x 4 = 16, which falls under the "High Risk" category. This risk is severe and could potentially cause 
            financial loss for Apple. The risk of international customers facing their second successive major price 
            increase has a likelihood rating of "Likely" and an impact rating of "Moderate". Therefore, the risk score 
            for this risk is 4 x 3 = 12, which also falls under the "High Risk" category. This risk is significant and 
            could potentially damage Apple's reputation and customer loyalty.""", additional_kwargs={"name": "example_assistant"})
            human_template = """Please score the risks for the company {company} as identified in this risk 
            assessment: {risk_analysis}. Your final answer should include a detailed explanation for your reasoning."""
            human_message_prompt: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(human_template)
            chat_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [system_message_prompt, example_message_human, example_message_ai, human_message_prompt]
            )
            llm: ChatOpenAI = ChatOpenAI(
                model="gpt-4",
                temperature = 0, 
                client = chat_prompt, 
                openai_api_key = config_secrets.read_openai_credentials(),
                verbose = True
            )
            agent: AgentExecutor = initialize_agent(
                tools = tools_risk_scoring_system,
                llm = llm, 
                agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
                verbose = True
            )
            risk_types_severity: str = agent.run(chat_prompt.format_messages(company = company, risk_analysis = risk_analysis))
            logging.info("Done awarding risks.")

            self.combined_result = "Keypoints:\n\n" + keypoints + "\n\n" + "Analysis:\n\n" + risk_analysis + "Risk Types Severity:\n\n" + risk_types_severity
        except (ValueError, TypeError) as e:
            if message_type and message_type == "analysis":
                db.close()
            logging.error(e)
            raise ValueError(f"Error: {e}") from e
        if message_type and message_type == "analysis":
            db.close()
        return self.combined_result