import autogen
from src.autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from ..autogen import AssistantAgent, UserProxyAgent
import chromadb
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("API Key not found!")

openai.api_key = api_key


class GroupChatManager:
    def __init__(self):
        self.llm_config = self.get_llm_config(api_key)
        self.agents = self.initialize_agents()

    @staticmethod
    def get_llm_config(api_key: str):
        return {
            "request_timeout": 60,
            "seed": 42,
            "config_list": [{'model': 'gpt-3.5-turbo', 'api_key': api_key}],
            "temperature": 0,
        }

    def initialize_agents(self):
        def termination_msg(x): return isinstance(
            x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

        self.boss = autogen.UserProxyAgent(
            name="Boss",
            is_termination_msg=termination_msg,
            human_input_mode="TERMINATE",
            system_message="The boss who asks questions and gives tasks, comes back for user feedback every 3 rounds",
            code_execution_config=False,
        )

        self.boss_aid = RetrieveUserProxyAgent(
            name="Boss_Assistant",
            is_termination_msg=termination_msg,
            system_message="Assistant who has extra content retrieval power for solving difficult problems.",
            human_input_mode="TERMINATE",
            max_consecutive_auto_reply=3,
            retrieve_config={
                "task": "code",
                "docs_path": "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
                "chunk_token_size": 1000,
                "model": self.llm_config['config_list'][0]["model"],
                "client": chromadb.PersistentClient(path="/tmp/chromadb"),
                "collection_name": "groupchat",
                "get_or_create": True,
            },
            code_execution_config=False,
        )

        self.coder = AssistantAgent(
            name="Senior_Python_Engineer",
            is_termination_msg=termination_msg,
            system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
            code_execution_config={
                "work_dir": "coding",
                "use_docker": True
            },
            llm_config=self.llm_config,
        )

        self.pm = AssistantAgent(
            name="Product_Manager",
            is_termination_msg=termination_msg,
            system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
            llm_config=self.llm_config,
        )

        self.reviewer = AssistantAgent(
            name="Code_Reviewer",
            is_termination_msg=termination_msg,
            system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
            llm_config=self.llm_config,
        )

        return [self.boss, self.boss_aid, self.coder, self.pm, self.reviewer]

    def reset_agents(self):
        for agent in self.agents:
            agent.reset()

    def run_flow(self, prompt: str, flow: str = "default") -> None:
        self.reset_agents()
        if flow == "rag_chat":
            return self.rag_chat(prompt)
        elif flow == "norag_chat":
            return self.norag_chat(prompt)
        elif flow == "call_rag_chat":
            return self.call_rag_chat(prompt)
        else:
            raise ValueError(f"Unsupported flow: {flow}")

    def rag_chat(self, prompt: str):
        self.reset_agents()
        groupchat = autogen.GroupChat(
            agents=[self.boss_aid, self.coder, self.pm,
                    self.reviewer], messages=[], max_round=12
        )
        manager = autogen.GroupChatManager(
            groupchat=groupchat, llm_config=self.llm_config)

        # Start chatting with boss_aid as this is the user proxy agent.
        self.boss_aid.initiate_chat(
            manager,
            problem=prompt,
            n_results=3,
        )

        return groupchat.messages

    def norag_chat(self, prompt: str):
        self.reset_agents()
        groupchat = autogen.GroupChat(
            agents=[self.boss, self.coder, self.pm,
                    self.reviewer], messages=[], max_round=12
        )
        manager = autogen.GroupChatManager(
            groupchat=groupchat, llm_config=self.llm_config)

        # Start chatting with boss as this is the user proxy agent.
        self.boss.initiate_chat(
            manager,
            message=prompt,
        )

        return groupchat.messages

    def call_rag_chat(self, prompt: str):
        self.reset_agents()
        # In this case, we will have multiple user proxy agents and we don't initiate the chat
        # with RAG user proxy agent.
        # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
        # it from other agents.

        def retrieve_content(message, n_results=3):
            # Set the number of results to be retrieved.
            self.boss_aid.n_results = n_results
            # Check if we need to update the context.
            update_context_case1, update_context_case2 = self.boss_aid._check_update_context(
                message)
            if (update_context_case1 or update_context_case2) and self.boss_aid.update_context:
                self.boss_aid.problem = message if not hasattr(
                    self.boss_aid, "problem") else self.boss_aid.problem
                _, ret_msg = self.boss_aid._generate_retrieve_user_reply(
                    message)
            else:
                ret_msg = self.boss_aid.generate_init_message(
                    message, n_results=n_results)
            return ret_msg if ret_msg else message

        # Disable human input for boss_aid since it only retrieves content.
        self.boss_aid.human_input_mode = "NEVER"

        llm_config = {
            "functions": [
                {
                    "name": "retrieve_content",
                    "description": "retrieve content for code generation and question answering.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
                            }
                        },
                        "required": ["message"],
                    },
                },
            ],
            "config_list": self.config_list,
            "request_timeout": 60,
            "seed": 42,
        }

        for agent in [self.coder, self.pm, self.reviewer]:
            # update llm_config for assistant agents.
            agent.llm_config.update(llm_config)

        for agent in [self.boss, self.coder, self.pm, self.reviewer]:
            # register functions for all agents.
            agent.register_function(
                function_map={
                    "retrieve_content": retrieve_content,
                }
            )

        groupchat = autogen.GroupChat(
            agents=[self.boss, self.coder, self.pm,
                    self.reviewer], messages=[], max_round=12
        )
        manager = autogen.GroupChatManager(
            groupchat=groupchat, llm_config=llm_config)

        # Start chatting with boss as this is the user proxy agent.
        self.boss.initiate_chat(
            manager,
            message=prompt,
        )

        return groupchat.messages
