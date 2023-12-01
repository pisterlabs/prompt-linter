"""
File that contains the logic for the feedback loop.
"""
from typing import Literal

from langchain.agents import AgentType, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain import SerpAPIWrapper
from langchain.utilities import WikipediaAPIWrapper

from src.logic.langchain_tools.tool_process_thought import process_thoughts

from src.logic.config import secrets as config_secrets

from db.database_connector import DatabaseConnector

class FeedbackLoop():
    """
    FeedbackLoop is a class that implements a feedback loop for a given prompt.
    """

    def __init__(self) -> None:
        pass

    def createDBConnection(self, db_name: str = "risk.db") -> DatabaseConnector:
        """
        Open a connection to the database and return a connection object.
        """
        db = DatabaseConnector(db_name)
        db.open()
        return db

    def human_input(self, company: str, problem: str, output: str) -> str:
        wikipedia: WikipediaAPIWrapper = WikipediaAPIWrapper()
        search: SerpAPIWrapper = SerpAPIWrapper(serpapi_api_key = config_secrets.read_serpapi_credentials())
        template_suggestions: Literal = """As a company specialist, you are tasked with answering questions about a company. This
        information will be used for a risk analysis. Your answers should be in line with the feedback given on a previous
        iteration of the risk analysis.
        """
        questions: str = output
        try:
            questions = output.split("Questions:")[1].strip()
            print(questions)
        except IndexError:
            print("No questions found.")
        system_message_prompt_suggestions = SystemMessagePromptTemplate.from_template(template_suggestions)
        human_template_suggestions: Literal = """The company in question is: {company}. The user feedback is: {problem} and the questions to answer are: {suggestions}."""
        human_message_prompt_suggestions: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(human_template_suggestions)
        chat_prompt_suggestions: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [system_message_prompt_suggestions, human_message_prompt_suggestions]
        )
        llm_suggestions = ChatOpenAI(
            model="gpt-4",
            temperature = 0,
            client = chat_prompt_suggestions,
            openai_api_key = config_secrets.read_openai_credentials(),
        )
        tools_suggestions = [
            Tool(
                name = "Search",
                func = search.run,
                description = "useful for when you need to answer questions about current events"
            ),
            Tool(
                name = "Thought Processing",
                func = process_thoughts,
                description = """useful for when you have a thought that you want to use in a task,
                but you want to make sure it's formatted correctly"""
            ),
            Tool(
                name = "Wikipedia",
                func = wikipedia.run,
                description = "useful for when you need to detailed information about a topic"
            )
        ]
        agent_suggestions: AgentExecutor = initialize_agent(
            tools = tools_suggestions, llm = llm_suggestions, agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose = True
        )
        suggestions_answer = agent_suggestions.run(
            chat_prompt_suggestions.format_prompt(company = company, problem = problem, suggestions = questions).to_messages()
        )
        return suggestions_answer

    def initialize_chain(self, instructions, memory=None):
        if memory is None:
            memory = ConversationBufferWindowMemory()
            memory.ai_prefix = "Assistant"
        template = f"""
        Instructions: {instructions}
        {{{memory.memory_key}}}
        Human: {{human_input}}
        Assistant:"""

        prompt = PromptTemplate(
            input_variables=["history", "human_input"], template=template
        )
        chain = LLMChain(
            llm = ChatOpenAI(model="gpt-4", temperature=0),
            prompt = prompt,
            verbose = True,
            memory = ConversationBufferWindowMemory(),
        )
        return chain
    
    def initialize_meta_chain(self):
        meta_template = """
        Assistant has just had the below interactions with a User. Assistant followed their "Instructions" closely. Your job is to critique the Assistant's performance and then revise the Instructions so that Assistant would quickly and correctly respond in the future.
        ####
        {chat_history}
        ####
        Please reflect on these interactions.   
        You should first critique Assistant's performance. What could Assistant have done better? What should the Assistant remember about this user? Are there things this user always wants? Indicate this with "Critique: ...".
        You should next revise the Instructions so that Assistant would quickly and correctly respond in the future. Assistant's goal is to satisfy the user in as few interactions as possible. Assistant will only see the new Instructions, not the interaction history, so anything important must be summarized in the Instructions. Don't forget any important details in the current Instructions! Indicate the new Instructions by "Instructions: ...".
        Remember that the goal is to optimize the user's prompt.
        """
        meta_prompt = PromptTemplate(
            input_variables=["chat_history"], template=meta_template
        )
        meta_chain = LLMChain(
            llm = ChatOpenAI(model="gpt-4", temperature=0),
            prompt = meta_prompt,
            verbose = True,
        )
        return meta_chain

    def get_chat_history(self, chain_memory):
        memory_key = chain_memory.memory_key
        chat_history = chain_memory.load_memory_variables(memory_key)[memory_key]
        return chat_history

    def get_new_instructions(self, meta_output):
        delimiter = "Instructions: "
        new_instructions = meta_output[meta_output.find(delimiter) + len(delimiter) :]
        return new_instructions
    
    def main(self, task: str, company: str, problem: str, message_type:str, max_iters=3, max_meta_iters=1):
        failed_phrase = "task failed"
        success_phrase = "task succeeded"
        key_phrases = [success_phrase, failed_phrase]
        meta_task = """Act as a Prompt Creator. Your goal is to help optimize a prompt. You will follow the following process: 
        1. Your first response will be to ask what should be reworked about the current prompt. An answer will be provided.
        2. Based on the input, you will generate 2 sections. a) Revised prompt (provide your rewritten prompt. it should 
        be clear, concise, and easily understood by you), and b) Questions (ask any relevant questions pertaining to what 
        additional information is needed to improve the prompt)."""
        meta_task = meta_task + "\nPrompt:" + task
        instructions = ""
        for i in range(max_meta_iters):
            print(f"[Episode {i+1}/{max_meta_iters}]")
            chain = self.initialize_chain(instructions, memory = None)
            output = chain.predict(human_input = meta_task)
            for j in range(max_iters):
                print(f"(Step {j+1}/{max_iters})")
                print(f"Assistant: {output}")
                print(f"Human: ")
                if i == 0:
                    human_input = problem
                else:
                    human_input = self.human_input(company=company, problem=problem, output=output)
                if any(phrase in human_input.lower() for phrase in key_phrases):
                    break
                output = chain.predict(human_input=human_input)
            if j+1 == max_iters and i+1 == max_meta_iters:
                print(output)
                revised_prompt = ""
                if "revised prompt:" in output.lower():
                    revised_prompt = output.split("Revised prompt:")[1].strip()
                    if "questions:" in revised_prompt.lower():
                        revised_prompt = revised_prompt.split("b) Questions:")[0].strip()
                else:
                    revised_prompt = output.strip()
                db = self.createDBConnection()
                db.c.execute("INSERT INTO prompts (message, type) VALUES (?, ?)", (revised_prompt, message_type))
                db.conn.commit()
                return revised_prompt
            meta_chain = self.initialize_meta_chain()
            meta_output = meta_chain.predict(chat_history = self.get_chat_history(chain.memory))
            print(f"Feedback: {meta_output}")
            instructions = self.get_new_instructions(meta_output)
            print(f"New Instructions: {instructions}")
            print("\n" + "#" * 80 + "\n")
        print(f"Let's enter the next episode!")