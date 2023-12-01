import asyncio
import json
import math
import os
from collections import deque
from itertools import islice
from typing import AsyncIterable, Iterable, Optional

import networkx as nx
import nltk
import numpy as np
import numpy.typing as npt
import openai
import pandas as pd
import tqdm
import typer
from asyncstdlib import map as amap
from asyncstdlib.functools import reduce as areduce
from graphviz import Digraph
from instructor.patch import wrap_chatcompletion
from pydantic import BaseModel, Field
from tenacity import retry, wait_random_exponential

from prompting_techniques import AsyncTyper, async_disk_cache, execute, format_prompt

np.random.seed(1)

nltk.download("punkt")

client = openai.AsyncOpenAI()
app = AsyncTyper()
func = wrap_chatcompletion(client.chat.completions.create)


class VectorDatabase(BaseModel):
    text: list[str]
    embeddings: npt.NDArray[np.float32]

    class Config:
        arbitrary_types_allowed = True

    def save_to_file(self, filename: str):
        # Convert NumPy array to a list for JSON serialization
        data = {"text": self.text, "embeddings": self.embeddings.tolist()}
        with open(filename, "w") as file:
            json.dump(data, file)

    @classmethod
    def load_from_file(cls, filename: str):
        with open(filename, "r") as file:
            data = json.load(file)
        # Convert list back to NumPy array
        data["embeddings"] = np.array(data["embeddings"], dtype=np.float32)
        return cls(**data)

    async def add_text(self, text: str) -> None:
        embeddings_response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        embedding: npt.NDArray[np.float32] = np.expand_dims(
            np.array(embeddings_response.data[0].embedding), axis=0
        )
        self.text.append(text)
        self.embeddings = np.concatenate([self.embeddings, embedding], axis=0)

    async def top_k(self, query: str, k: int = 10) -> list[str]:
        query_embedding_response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=query,
        )
        query_embedding: npt.NDArray[np.float32] = np.array(
            query_embedding_response.data[0].embedding
        )

        # cosine similarity
        similarity: npt.NDArray[np.float32] = np.dot(query_embedding, self.embeddings.T) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(self.embeddings, axis=1)
        )
        sorted_similarity_indices: npt.NDArray[np.int64] = np.argsort(similarity)[::-1]
        top_k: list[str] = [self.text[i] for i in sorted_similarity_indices[:k]]
        return top_k
    

class TaskNames(BaseModel):
    names: list[str]

class Task(BaseModel):
    task_id: int = Field(..., description="The unique identifier for the task.")
    task_name: str = Field(..., description="The name of the task.")

class TaskResult(BaseModel):
    task_name: str = Field(..., description="The name of the task.")
    task_result: str = Field(..., description="The result of the task.")

class SingleTaskListStorage:
    def __init__(self):
        self.tasks: deque[Task] = deque([])
        self.task_id_counter = 0

    def append(self, task: Task):
        self.tasks.append(task)

    def replace(self, tasks: list[Task]):
        self.tasks = deque(tasks)

    def popleft(self) -> Task:
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self) -> int:
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t.task_name for t in self.tasks]




@retry(wait=wait_random_exponential(multiplier=1, max=3))
async def execution_agent(vecdb: VectorDatabase, objective: str, task: Task, k: int = 5) -> TaskResult:
    related_entries = await vecdb.top_k(task.task_name, k=k)
    result = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": format_prompt(
                f"""
                You are a baby artificial general intelligence (AGI) system. You role is to execute tasks in pursuit of a goal.
                
                You will perform one task based on the objective: {objective}.
                
                Take into account previous relevant tasks you have performed: {related_entries}.
                """
                ),
            },
            {
                "role": "user",
                "content": format_prompt(
                f"""
                Please output your response to the following task: {task.task_name}

                Be as direct as possible and do not provide any other information.
                """
                ),
            },
        ],
        model="gpt-4",
        temperature=0,
        seed=256,
        max_tokens=128,
    )
    assert len(result.choices) > 0, "No choices were provided."
    content = result.choices[0].message.content
    assert content is not None, "No content was provided."
    return TaskResult(task_name=task.task_name, task_result=content)

@retry(wait=wait_random_exponential(multiplier=1, max=3))
async def task_creation_agent(objective: str, task_result: TaskResult, task_list: SingleTaskListStorage, max_new_tasks: int = 3) -> TaskNames:
    result: Optional[TaskNames] = None
    for _ in range(3):
        try:
            messages=[
                {
                    "role": "system",
                    "content": format_prompt(
                    f"""
                    You are a baby artificial general intelligence (AGI) system. You role is to execute tasks in pursuit of a goal.
                    
                    You are to use the result from an execution agent to create new tasks with the following objective: {objective}.
                    The last completed task has the result: {task_result.model_dump_json()}
                    This result was based on this task description: {task_result.task_name}.
                    """
                    ),
                },
            ]
            if not task_list.is_empty():
                messages.append({
                    "role": "system",
                    "content": format_prompt(
                    f"""
                    You have the following incomplete tasks in your task list: {task_list.get_task_names()}
                    """
                    ),
                })
            messages.append({
                "role": "user",
                "content": format_prompt(
                f"""
                Add a list of tasks to your task list. Each task should be on a new line and not conflict with the objective or other tasks.
                
                If no tasks need to be added, just output an empty list.
                """
                ),
            })
            
            result = await asyncio.wait_for(
                func(
                    messages=messages,
                    model="gpt-3.5-turbo-0613",
                    response_model=TaskNames,
                    temperature=0.1,
                    seed=256,
                ),
                timeout=30,
            )
            break
        except asyncio.TimeoutError:
            continue
    if result is None:
        raise RuntimeError("Failed to classify article after 3 attempts")
    return result

@retry(wait=wait_random_exponential(multiplier=1, max=3))
async def task_priority_agent(objective: str, task_list: SingleTaskListStorage) -> TaskNames:
    result: Optional[TaskNames] = None
    tasks_prompt = "\n".join(task_list.get_task_names())
    for _ in range(3):
        try:
            messages=[
                {
                    "role": "system",
                    "content": format_prompt(
                    f"""
                    You are a baby artificial general intelligence (AGI) system. You role is to execute tasks in pursuit of a goal.
                    
                    You are tasked with prioritizing the following tasks:
                    {tasks_prompt}

                    Consider the ultimate objective of your team: {objective}.
                    
                    Tasks should be sorted from highest to lowest priority, where higher-priority tasks are those that act as pre-requisites or are more essential for meeting the objective.
                    Do not remove any tasks. Return the ranked tasks in the order of priority:
                    """
                    ),
                },
            ]            
            result = await asyncio.wait_for(
                func(
                    messages=messages,
                    model="gpt-3.5-turbo-0613",
                    response_model=TaskNames,
                    temperature=0.1,
                    seed=256,
                ),
                timeout=30,
            )
            break
        except asyncio.TimeoutError:
            continue
    if result is None:
        raise RuntimeError("Failed to classify article after 3 attempts")
    return result

@retry(wait=wait_random_exponential(multiplier=1, max=3))
async def best_breakfast_result(objective: str, task_results: list[TaskResult]) -> AsyncIterable[str]:
    tasks_prompt = "\n".join(map(lambda t: t.model_dump_json(), task_results))
    result = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": format_prompt(
                f"""
                You are a baby artificial general intelligence (AGI) system. You role is to execute tasks in pursuit of a goal.
                
                You have just executed a series of tasks in pursuit of an objective: {objective}.
                
                Here are your task results:
                {tasks_prompt}
                """
                ),
            },
            {
                "role": "user",
                "content": format_prompt(
                f"""
                Based on the task results, answer the objective: {objective}.
                
                Output a single distinctive answer that is as short as possible.
                """
                ),
            }
        ],
        max_tokens=64,
        temperature=0.9,
        model="gpt-4",
        stream=True,   
    )
    async for message in result:
        assert len(message.choices) > 0, "No translation was provided."
        content = message.choices[0].delta.content
        if content is not None:
            yield content

async def read_or_create_vecdb(vecdb_filename) -> VectorDatabase:
    if os.path.exists(vecdb_filename):
        vecdb = VectorDatabase.load_from_file(vecdb_filename)
    else:
        vecdb = VectorDatabase(text=[], embeddings=np.empty((0, 1536), dtype=np.float32))
        vecdb.save_to_file(vecdb_filename)
    return vecdb


@app.command()
async def vec_lookup(n: int = 8):
    
    ## Load vecdb
    vecdb_filename = "./data/vecdb.json"
    vecdb = await read_or_create_vecdb(vecdb_filename)
    
    ## Collect user input task
    objective: str = str(typer.prompt(f"BabyAGI objective", type=str))
    assert len(objective) > 0, "Please provide some text."
    task: str = str(typer.prompt(f"BabyAGI task", type=str))
    assert len(task) > 0, "Please provide some text."
    
    ## Add first task to the task list
    tasks = SingleTaskListStorage()
    initial_task = Task(task_id=tasks.next_task_id(), task_name=task)
    tasks.append(initial_task)
    
    ## history
    history: list[TaskResult] = []

    ## Run babyagi for n stepstoken
    for step in range(n):
        if tasks.is_empty():
            typer.echo(f"Done!")
        else:
            typer.echo("\n")
            typer.echo(f"*** Step {step} ***")

            ## Log the task list and current task            
            typer.echo(f"Task list:")
            for name in tasks.get_task_names():
                typer.echo(f"  - {name}")
            current_task = tasks.popleft()
            typer.echo(f"Current task: {current_task.task_name}")
            
            ## Execute the current task
            current_task_result = await execution_agent(vecdb, objective, current_task)
            await vecdb.add_text(current_task_result.model_dump_json())
            history.append(current_task_result)
            typer.echo(f"Current tasks result: {current_task_result.task_result}")            
            
            ## Create new tasks 
            new_task_names = await task_creation_agent(
                objective, current_task_result, tasks
            )
            for new_task_name in new_task_names.names:
                tasks.append(Task(task_id=tasks.next_task_id(), task_name=new_task_name))

            ## Prioritize tasks
            new_task_names = await task_priority_agent(objective, tasks)
            tasks = SingleTaskListStorage()
            for new_task_name in new_task_names.names:
                tasks.append(Task(task_id=tasks.next_task_id(), task_name=new_task_name))
    
    typer.echo(f"History:")
    max_str_len = 50
    for task_result in history:
        if len(task_result.task_result) > max_str_len:
            typer.echo(f"  - {task_result.task_name}: {task_result.task_result[:max_str_len]}...")
        else:
            typer.echo(f"  - {task_result.task_name}: {task_result.task_result}")

    typer.echo("\n")
    typer.echo("\n")
    typer.echo("\n")
    typer.echo(f"Objective result:")            
    async for token in best_breakfast_result(objective, history):
        typer.echo(token, nl=False)

if __name__ == "__main__":
    app()
