"""BabyAGP Plugin with Langchain"""
import firebase_admin
from firebase_admin import db
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental import BabyAGI

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.callbacks.manager import CallbackManagerForChainRun
import faiss

from Brain.src.rising_plugin.llm.llms import (
    MAX_AUTO_THINKING,
    get_finish_command_for_auto_task,
)


class BabyAGILLM:
    """BabyAGI run method to get the expected result"""

    def run(
        self,
        agent: BabyAGI,
        inputs: Dict[str, Any],
        firebase_app: firebase_admin.App,
        reference_link: str,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """firebase realtime database init"""
        ref = db.reference(reference_link, app=firebase_app)

        """Run the agent."""
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a todo list")
        agent.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if agent.task_list:
                agent.print_task_list()

                # Step 1: Pull the first task
                task = agent.task_list.popleft()
                agent.print_next_task(task)
                # update the result with the task in firebase realtime database
                ref.push().set(task)

                # Step 2: Execute the task
                result = agent.execute_task(objective, task["task_name"])
                this_task_id = int(task["task_id"])
                agent.print_task_result(result)
                # add result of the command
                ref.push().set({"result": result})

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                agent.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = agent.get_next_task(result, task["task_name"], objective)
                for new_task in new_tasks:
                    agent.task_id_counter += 1
                    new_task.update({"task_id": agent.task_id_counter})
                    agent.add_task(new_task)
                agent.task_list = deque(agent.prioritize_tasks(this_task_id, objective))
            num_iters += 1
            if (
                agent.max_iterations is not None and num_iters == agent.max_iterations
            ) or num_iters == MAX_AUTO_THINKING:
                print(
                    "\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m"
                )
                # add finish command of the command
                ref.push().set(get_finish_command_for_auto_task())
                break
        return {}

    def ask_task(
        self, query: str, firebase_app: firebase_admin.App, reference_link: str
    ):
        # Define your embedding model
        embeddings_model = OpenAIEmbeddings()

        # Initialize the vectorstore as empty
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query, index, InMemoryDocstore({}), {}
        )

        llm = OpenAI(temperature=0)

        # Logging of LLMChains
        verbose = False
        # If None, will keep on going forever
        max_iterations: Optional[int] = 3
        baby_agi = BabyAGI.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            verbose=verbose,
            max_iterations=max_iterations,
        )

        # querying
        self.run(
            agent=baby_agi,
            inputs={"objective": query},
            firebase_app=firebase_app,
            reference_link=reference_link,
        )
