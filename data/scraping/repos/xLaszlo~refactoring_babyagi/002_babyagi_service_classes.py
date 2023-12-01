# Strip comments and prints
# Declare three service classes: openai, pinecone, babyagi
# Move parameters into their constructors
# Declare basic setup in __main__
# Move loop into babyagi class
# Add dotenv and typer
# fix when a constant was moved to be a member variable
# Move get_ada_embedding() to OpenAIService
# Move agent functions into babyagi class
# Change infinite loop to fixed length for testing

import os
from dotenv import load_dotenv
import typer
import openai
import pinecone
from collections import deque


class OpenAIService:
    def __init__(self, api_key):
        openai.api_key = api_key

    def get_ada_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


class PineconeService:
    def __init__(self, api_key, environment, table_name, dimension, metric, pod_type):
        self.table_name = table_name
        pinecone.init(api_key=api_key, environment=environment)
        if table_name not in pinecone.list_indexes():
            pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        self.index = pinecone.Index(table_name)


class BabyAGI:
    def __init__(self, openai_service, pinecone_service):
        self.openai_service = openai_service
        self.pinecone_service = pinecone_service
        self.task_list = deque([])
        self.objective = None

    def add_task(self, task):
        self.task_list.append(task)

    def task_creation_agent(objective, result, task_description, task_list):
        prompt = f'You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}, The last completed task has the result: {result}. This result was based on this task description: {task_description}. These are incomplete tasks: {", ".join(task_list)}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array.'
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        new_tasks = response.choices[0].text.strip().split("\n")
        return [{"task_name": task_name} for task_name in new_tasks]

    def prioritization_agent(self, this_task_id):
        global task_list
        task_names = [t["task_name"] for t in task_list]
        next_task_id = int(this_task_id) + 1
        prompt = f"""You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}. Consider the ultimate objective of your team:{self.objective}. Do not remove any tasks. Return the result as a numbered list, like:
        #. First task
        #. Second task
        Start the task list with number {next_task_id}."""
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        new_tasks = response.choices[0].text.strip().split("\n")
        task_list = deque()
        for task_string in new_tasks:
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                task_list.append({"task_id": task_id, "task_name": task_name})

    def execution_agent(self, objective, task):
        context = self.context_agent(index=self.pinecone_service.table_name, query=objective, n=5)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"You are an AI who performs one task based on the following objective: {objective}. Your task: {task}\nResponse:",
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].text.strip()

    def context_agent(self, query, index, n):
        query_embedding = self.openai_service.get_ada_embedding(query)
        index = pinecone.Index(index_name=index)
        results = index.query(query_embedding, top_k=n, include_metadata=True)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata["task"])) for item in sorted_results]

    def run(self, objective, first_task):
        self.add_task({"task_id": 1, "task_name": first_task})
        self.objective = objective
        task_id_counter = 1
        for _ in range(5):
            if task_list:
                for t in task_list:
                    print(str(t["task_id"]) + ": " + t["task_name"])
                task = task_list.popleft()
                result = self.execution_agent(self.objective, task["task_name"])
                this_task_id = int(task["task_id"])
                enriched_result = {"data": result}
                result_id = f'result_{task["task_id"]}'
                vector = enriched_result["data"]
                self.pinecone_service.index.upsert(
                    [
                        (
                            result_id,
                            self.openai_service.get_ada_embedding(vector),
                            {"task": task["task_name"], "result": result},
                        )
                    ]
                )
            new_tasks = self.task_creation_agent(
                self.objective,
                enriched_result,
                task["task_name"],
                [t["task_name"] for t in task_list],
            )

            for new_task in new_tasks:
                task_id_counter += 1
                new_task.update({"task_id": task_id_counter})
                self.add_task(new_task)
            self.prioritization_agent(this_task_id)


def main():
    load_dotenv()
    print(os.getenv("PINECONE_API_KEY"))
    baby_agi = BabyAGI(
        openai_service=OpenAIService(api_key=os.getenv("OPENAI_API_KEY")),
        pinecone_service=PineconeService(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT"),
            table_name="test-table",
            dimension=1536,
            metric="cosine",
            pod_type="p1",
        ),
    )
    baby_agi.run(objective="Solve world hunger.", first_task="Develop a task list.")


if __name__ == "__main__":
    typer.run(main)
