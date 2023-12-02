"""autogpt plugin with langchain"""
import json

import firebase_admin
from firebase_admin import db
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import (
    AIMessage,
    Document,
    HumanMessage,
    SystemMessage,
)
from typing import List

from marshmallow import ValidationError

import faiss

from Brain.src.rising_plugin.llm.llms import (
    MAX_AUTO_THINKING,
    get_finish_command_for_auto_task,
)


class AutoGPTLLM:
    """autogpt run method to get the expected result"""

    def run(
        self,
        agent: AutoGPT,
        goals: List[str],
        firebase_app: firebase_admin.App,
        reference_link: str,
    ) -> str:
        """firebase realtime database init"""
        ref = db.reference(reference_link, app=firebase_app)

        """autogpt engine"""
        user_input = (
            "Determine which next command to use, "
            "and respond using the format specified above:"
        )
        # Interaction Loop
        loop_count = 0
        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1

            # validation thinking counter
            if loop_count == MAX_AUTO_THINKING:
                # add finish command of the command
                ref.push().set(get_finish_command_for_auto_task())
                break

            # Send message to AI, get response
            assistant_reply = agent.chain.run(
                goals=goals,
                messages=agent.full_message_history,
                memory=agent.memory,
                user_input=user_input,
            )
            # update the result with the assistant_reply in firebase realtime database
            ref.push().set(json.loads(assistant_reply))

            # update chat history in autogpt agent
            # Print Assistant thoughts
            print(assistant_reply)
            agent.full_message_history.append(HumanMessage(content=user_input))
            agent.full_message_history.append(AIMessage(content=assistant_reply))

            # Get command name and arguments
            action = agent.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in agent.tools}
            if action.name == FINISH_NAME:
                return action.args["response"]
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                result = f"Command {tool.name} returned: {observation}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            )
            if agent.feedback_tool is not None:
                feedback = f"\n{agent.feedback_tool.run('Input: ')}"
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    return "EXITING"
                memory_to_add += feedback

            agent.memory.add_documents([Document(page_content=memory_to_add)])
            agent.full_message_history.append(SystemMessage(content=result))
            # add result of the command
            ref.push().set({"result": result})

    """function to manage auto-task achievement
    ex: query = write a weather report for SF today
    """

    def ask_task(
        self, query: str, firebase_app: firebase_admin.App, reference_link: str
    ):
        search = SerpAPIWrapper()
        tools = [
            Tool(
                name="search",
                func=search.run,
                description="useful for when you need to answer questions about current events. You should ask targeted questions",
            ),
            WriteFileTool(),
            ReadFileTool(),
        ]

        # Define your embedding model
        embeddings_model = OpenAIEmbeddings()
        # Initialize the vectorstore as empty

        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query, index, InMemoryDocstore({}), {}
        )

        agent = AutoGPT.from_llm_and_tools(
            ai_name="Tom",
            ai_role="Assistant",
            tools=tools,
            llm=ChatOpenAI(temperature=0),
            memory=vectorstore.as_retriever(),
        )
        # Set verbose to be true
        agent.chain.verbose = True
        self.run(
            agent=agent,
            goals=[query],
            firebase_app=firebase_app,
            reference_link=reference_link,
        )
