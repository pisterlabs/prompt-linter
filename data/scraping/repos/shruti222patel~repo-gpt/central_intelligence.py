import logging

from repo_gpt.agents.base_agent import BaseAgent
from repo_gpt.agents.code_writer import CodeWritingAgent
from repo_gpt.agents.repo_comprehender import RepoUnderstandingAgent
from repo_gpt.file_handler.generic_code_file_handler import PythonFileHandler
from repo_gpt.openai_service import OpenAIService
from repo_gpt.search_service import SearchService

logger = logging.getLogger(__name__)


class CentralIntelligenceAgent(BaseAgent):
    system_prompt = """You are an expert software engineer. You have a few helper agents that help you understand and write good software. You can call these agents by using the following functions:
    - understand_the_codebase_and_formulate_plan(query): Use this function to call an LLM agent to understand the codebase and formulate a plan of what files need to be updated and how they need to be updated. Also use this function to answer general questions about the codebase. The input should be a query about the codebase.
    - update_code(plan): Use this function to call an LLM agent to update the code in the repository. The input should be a plan of what files need to be updated and how they need to be updated.
Use the two llm agents to complete the user task. Always understand the codebase first and follow the existing coding practices
**DO NOT** respond to the user directly. Use the functions instead.
"""

    def __init__(
        self,
        user_task,
        root_path,
        embedding_file_path,
        threshold=10 * 2,
        debug=False,
        openai_key=None,
    ):
        system_prompt = "You are an expert software engineer writing code in a repository. The user gives you a plan detailing how the code needs to be updated. You implement the code changes using functions. Ask clarifying questions."
        super().__init__(
            user_task, "completed_all_code_updates", system_prompt, threshold, debug
        )  # Call ParentAgent constructor
        self.root_path = root_path
        self.embedding_path = embedding_file_path
        self.openai_key = openai_key
        self.openai_service = (
            OpenAIService() if not openai_key else OpenAIService(openai_key)
        )
        self.functions = self._initialize_functions()

    def _initialize_functions(self):
        return [
            {
                "name": "understand_the_codebase_and_formulate_plan",
                "description": "Use this function to call an LLM agent to understand the codebase and formulate a plan of what files need to be updated and how they need to be updated. Also use this function to answer general questions about the codebase. The input should be a query about the codebase.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The task that needs to be accomplished or a general repository question that must be answered.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "update_code",
                "description": "Use this function to call an LLM agent to update the code in the repository. The input should be a plan of what files need to be updated and how they need to be updated.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "string",
                            "description": "A detailed plan of what files need to be updated and how they need to be updated.",
                        }
                    },
                    "required": ["plan"],
                },
            },
            {
                "name": "users_task_is_completed",
                "description": "Call this function when the user's task is completed. ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary_of_actions_taken": {
                            "type": "string",
                            "description": "Enumeration of all the changes that were made to the code.",
                        }
                    },
                    "required": ["summary_of_actions_taken"],
                },
            },
        ]

    def understand_the_codebase_and_formulate_plan(self, query):
        repo_agent = RepoUnderstandingAgent(
            query,
            self.root_path,
            self.embedding_path,
            openai_key=self.openai_key,
            debug=True,
        )
        return repo_agent.process_messages()

    def update_code(self, plan):
        writer_agent = CodeWritingAgent(
            plan,
            self.root_path,
            self.embedding_path,
            openai_key=self.openai_key,
            debug=True,
        )
        return writer_agent.process_messages()

    def users_task_is_completed(self, summary_of_actions_taken):
        return summary_of_actions_taken
