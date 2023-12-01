# Move OpenAI calls into the service

import pickle
import numpy as np
import os
from dotenv import load_dotenv
import typer
import openai
import pinecone
from collections import deque
from collections import namedtuple


class OpenAIService:
    def __init__(self, api_key):
        openai.api_key = api_key

    def get_ada_embedding(self, text):
        print('get_ada_embedding')
        text = text.replace('\n', ' ')
        return openai.Embedding.create(input=[text], model='text-embedding-ada-002')['data'][0]['embedding']

    def create_tasks_task_creation_agent(self, objective, result, task_description, task_list):
        prompt = f'''
            You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
            The last completed task has the result: {result}. This result was based on this task description: {task_description}.
            These are incomplete tasks: {", ".join(task_list)}.
            Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
            Return the tasks as an array.'''
        return [
            {'task_name': task_name}
            for task_name in openai.Completion.create(
                engine='text-davinci-003',
                prompt=prompt,
                temperature=0.5,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            .choices[0]
            .text.strip()
            .split('\n')
        ]

    def create_tasks_prioritization_agent(self, this_task_id, objective, task_list):
        print('create_tasks_prioritization_agent')
        task_names = [t['task_name'] for t in task_list]
        prompt = f'''
            You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
            Consider the ultimate objective of your team:{objective}. Do not remove any tasks. Return the result as a numbered list, like:
            #. First task
            #. Second task
            Start the task list with number {int(this_task_id) + 1}.'''
        return (
            openai.Completion.create(
                engine='text-davinci-003',
                prompt=prompt,
                temperature=0.5,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            .choices[0]
            .text.strip()
            .split('\n')
        )

    def create_tasks_execution_agent(self, objective, task):
        print('create_tasks_execution_agent')
        prompt = f'''
            You are an AI who performs one task based on the following objective: {objective}. Your task: {task}\nResponse:'''
        return (
            openai.Completion.create(
                engine='text-davinci-003',
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            .choices[0]
            .text.strip()
        )


class CachedOpenAIService:
    def __init__(self, openai_service, cache_file):
        self.openai_service = openai_service
        self.cache_file = cache_file
        if os.path.isfile(cache_file):
            self.cache = pickle.load(open(cache_file, 'rb'))
        else:
            self.cache = {
                'get_ada_embedding': {},
                'create_tasks_task_creation_agent': {},
                'create_tasks_prioritization_agent': {},
                'create_tasks_execution_agent': {},
            }
            pickle.dump(self.cache, open(cache_file, 'wb'))

    def get_ada_embedding(self, text):
        if text not in self.cache['get_ada_embedding']:
            self.cache['get_ada_embedding'][text] = self.openai_service.get_ada_embedding(text)
            pickle.dump(self.cache, open(self.cache_file, 'wb'))
        return self.cache['get_ada_embedding'][text]

    def create_tasks_task_creation_agent(self, objective, result, task_description, task_list):
        key = (objective, str(result), task_description, str(task_list))
        chache_name = 'create_tasks_task_creation_agent'
        if key not in self.cache[chache_name]:
            result = self.openai_service.create_tasks_task_creation_agent(
                objective, result, task_description, task_list
            )
            self.cache[chache_name][key] = result
            pickle.dump(self.cache, open(self.cache_file, 'wb'))
        return self.cache[chache_name][key]

    def create_tasks_prioritization_agent(self, this_task_id, objective, task_list):
        key = (this_task_id, objective, str(task_list))
        chache_name = 'create_tasks_prioritization_agent'
        if key not in self.cache[chache_name]:
            result = self.openai_service.create_tasks_prioritization_agent(this_task_id, objective, task_list)
            self.cache[chache_name][key] = result
            pickle.dump(self.cache, open(self.cache_file, 'wb'))
        return self.cache[chache_name][key]

    def create_tasks_execution_agent(self, objective, task):
        key = (objective, task)
        chache_name = 'create_tasks_execution_agent'
        if key not in self.cache[chache_name]:
            result = self.openai_service.create_tasks_execution_agent(objective, task)
            self.cache[chache_name][key] = result
            pickle.dump(self.cache, open(self.cache_file, 'wb'))
        return self.cache[chache_name][key]


class PineconeService:
    def __init__(self, api_key, environment, table_name, dimension, metric, pod_type):
        self.table_name = table_name
        pinecone.init(api_key=api_key, environment=environment)
        if table_name not in pinecone.list_indexes():
            pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        self.index = pinecone.Index(table_name)

    def get_results(self, query_embedding, n):
        return self.index.query(query_embedding, top_k=n, include_metadata=True)

    def upsert(self, result_id, embedding, task, result):
        self.pinecone_service.index.upsert([(result_id, embedding, {'task': task['task_name'], 'result': result})])


class TestPineconeService:
    def __init__(self, api_key, environment, table_name, dimension, metric, pod_type):
        self.table_name = table_name
        self.dimension = dimension
        self.data = []
        self.vectors = None

    def get_results(self, query_embedding, n):
        Match = namedtuple('Match', 'score metadata')
        Result = namedtuple('Result', 'matches')
        if self.vectors is None:
            return Result([])
        score = self.vectors @ query_embedding
        return Result([Match(score[k], self.data[k]) for k in np.argsort(-score)[:n]])

    def upsert(self, result_id, embedding, task, result):
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        if self.vectors is None:
            self.vectors = embedding.reshape(1, -1)
            self.data = [{'result_id': result_id, 'task': task, 'result': result}]
        else:
            self.vectors = np.vstack((self.vectors, embedding))
            self.data.append({'result_id': result_id, 'task': task, 'result': result})


class BabyAGI:
    def __init__(self, openai_service, pinecone_service):
        self.openai_service = openai_service
        self.pinecone_service = pinecone_service
        self.task_list = deque([])
        self.objective = None

    def add_task(self, task):
        self.task_list.append(task)

    def prioritization_agent(self, this_task_id):
        task_list = deque()
        new_tasks = self.openai_service.create_tasks_prioritization_agent(this_task_id, self.objective, self.task_list)
        for task_string in new_tasks:
            task_parts = task_string.strip().split('.', 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                task_list.append({'task_id': task_id, 'task_name': task_name})

    def execution_agent(self, objective, task):
        context = self.context_agent(query=objective, n=5)
        return self.openai_service.create_tasks_execution_agent(objective, task)

    def context_agent(self, query, n):
        query_embedding = self.openai_service.get_ada_embedding(query)
        results = self.pinecone_service.get_results(query_embedding=query_embedding, n=n)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata['task'])) for item in sorted_results]

    def run(self, objective, first_task):
        self.add_task({'task_id': 1, 'task_name': first_task})
        self.objective = objective
        task_id_counter = 1
        for _ in range(2):
            if self.task_list:
                task = self.task_list.popleft()
                result = self.execution_agent(self.objective, task['task_name'])
                this_task_id = int(task['task_id'])
                enriched_result = {'data': result}
                result_id = f'result_{task["task_id"]}'
                vector = enriched_result['data']
                embedding = self.openai_service.get_ada_embedding(vector)
                self.pinecone_service.upsert(result_id, embedding, task, result)
            new_tasks = self.openai_service.create_tasks_task_creation_agent(
                self.objective,
                enriched_result,
                task['task_name'],
                [t['task_name'] for t in self.task_list],
            )
            for new_task in new_tasks:
                task_id_counter += 1
                new_task.update({'task_id': task_id_counter})
                self.add_task(new_task)
            self.prioritization_agent(this_task_id)


def main():
    load_dotenv()
    baby_agi = BabyAGI(
        openai_service=CachedOpenAIService(
            openai_service=OpenAIService(api_key=os.getenv('OPENAI_API_KEY')),
            cache_file='cache.pkl',
        ),
        pinecone_service=TestPineconeService(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT'),
            table_name='test-table',
            dimension=1536,
            metric='cosine',
            pod_type='p1',
        ),
    )
    baby_agi.run(objective='Solve world hunger.', first_task='Develop a task list.')


if __name__ == '__main__':
    typer.run(main)
