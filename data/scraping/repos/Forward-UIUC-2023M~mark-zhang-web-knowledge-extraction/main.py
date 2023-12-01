import os
import logging
import asyncio
from textwrap import dedent
from dotenv import load_dotenv
import pandas as pd
import guidance
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from duckduckgo_search import DDGS
from state_machine import State, Context
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
# LLM = guidance.llms.OpenAI("gpt-3.5-turbo")
LLM = guidance.llms.OpenAI("gpt-4")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    handlers=[
        logging.FileHandler("output.log"),
    ], 
    level=logging.INFO)


class EmbeddingStore:
    def __init__(self, model='text-embedding-ada-002'):
        self.store = pd.DataFrame(columns=['text', 'embedding'])
        self.model = model

    def add(self, text):
        embedding = get_embedding(text, engine=self.model)
        self.store[len(self.store.index)] = [text, embedding]

    def search(self, text, n=3):
        embedding = get_embedding(text, engine=self.model)
        self.store['similarities'] = self.store['embedding'].apply(lambda x: cosine_similarity(x, embedding))
        res = self.store.sort_values('similarities', ascending=False).head(n)['text']
        return res


def search(query: str, n=5):
    res = []
    for r in DDGS().text(query):
        res.append(r)
        if len(res) == n:
            break
    return res


def format_search_result(result):
    return f"# {result['title']}\n{result['href']}\n{result['body']}\n"


class CreatePlanState(State):
    def run(self):
        program = guidance(dedent("""\
            {{#system~}}
            You are a helpful agent
            {{~/system}}
            {{#user~}}
            Create a step-by-step search engine query plan that fulfills the given command by user.

            Command: Answer the question: Who wrote the play 'Hamlet,' and where was he born?
            Plan:
            1. search the author of 'Hamlet'
            2. edit the answer with the author
            3. search where the author was born in
            4. edit the answer with the birthplace
            5. terminate

            Command: Answer the question: Who are the current Prime Ministers of the UK and Canada, and who has a higher educational background?
            Plan:
            1. search for the Prime Minister of UK and navigate to result pages if necessary (wikipedia might be a useful site)
            2. update the answer with the education background of the Prime Minister of UK
            3. search for the Prime Minister of Canada and navigate to result pages if necessary (wikipedia)
            4. update the answer with the education background of the Prime Minister of Canada
            5. update the answer by comparing education backgrounds
            6. terminate

            Command: {{command}}
            Plan:
            {{~/user}}
            {{#assistant~}}
            {{gen 'plan' temperature=0.0 max_tokens=100}}
            {{~/assistant}}"""))
        results = program(command=self.context.command, llm=LLM)
        logging.info(results)
        self.context.plan = results['plan']
        self.context.transition_to(PreActionState())

class PreActionState(State):
    def run(self):
        program = guidance(dedent("""\
            {{#system~}}
            You are a helpful agent
            {{~/system}}
            {{#user~}}
            User command: {{command}}
            Current answer: {{current_answer}}
            Plan: {{plan}}
            Available Actions:
            - web-search [keywords] (search keywords using a search engine)
            - semantic-memory-search [text] (search for semantically similar text segments to answer question)
            - edit-answer [new-answer] (edit and update the current answer)
            - navigate [url] (navigate to the given url)
            - terminate
            Choose an action by outputing the action name followed by the action arguments. Please terminate if the answer is complete.
            {{~/user}}
            {{#assistant~}}
            {{gen 'action' temperature=0.1 max_tokens=150}}
            {{~/assistant}}"""))
        results = program(command=self.context.command,
                          plan=self.context.plan, 
                          current_answer=self.context.answer, 
                          llm=LLM)
        logging.info(results)
        action: str = results['action'].strip()
        self.context.actions.append(action)
        self.context.transition_to(ExecuteActionState())

class ExecuteActionState(State):
    def run(self):
        # parse and validate action
        action = self.context.actions[-1]
        # print action in red
        print(f"\033[91mAction: {action}\033[00m")
        if action.find('web-search') != -1:
            query = action.split('web-search')[1].strip()
            results = list(search(query))
            print("Result len: ", len(results))
            results = [format_search_result(res) for res in results]
            for result in results[:5]:
                self.context.embedding_store.add(result)
            self.context.transition_to(PostActionState('\n'.join(results[:3])))
        elif action.find('semantic-memory-search') != -1:
            query = action.split('semantic-memory-search')[1].strip()
            search_results = self.context.embedding_store.search(query, n=3)
            if len(search_results) == 0:
                self.context.transition_to(PostActionState("No results found"))
            else:
                result_strs = [str(result) for result in search_results]
                print(result_strs)
                self.context.transition_to(PostActionState(str('\n---\n'.join(result_strs))))
        elif action.find('edit-answer') != -1:
            answer = action.split('edit-answer')[1].strip()
            self.context.answer = answer
            self.context.transition_to(PostActionState("Answer edited succesfully"))
        elif action.find('terminate') != -1:
            self.context.transition_to(FinishedState())
        else:
            self.context.transition_to(PostActionState("Invalid action"))

class PostActionState(State):
    def __init__(self, action_results):
        super().__init__()   
        self.action_results = action_results

    def run(self):
        program = guidance(dedent("""\
            {{#system~}}
            You are a helpful agent
            {{~/system}}
            {{#user~}}
            User command: {{command}}
            Current answer: {{current_answer}}
            Plan: {{plan}}
            Action chosen: {{action}}
            Results: {{action_results}}
            Create an updated plan. Include in the plan the answer to compose for the next step if necessary. You do not need to "confirm" anything. Terminate if the answer is complete. Note: Avoid searching the same term repeatedly.
            {{~/user}}
            {{#assistant~}}
            {{gen 'new_plan' temperature=0.1 max_tokens=150}}
            {{~/assistant}}"""))
        results = program(command=self.context.command, 
                          plan=self.context.plan, 
                          current_answer=self.context.answer, 
                          action=self.context.actions[-1],
                          action_results=self.action_results,
                          llm=LLM)
        logging.info(results)
        self.context.plan = results['new_plan'].strip()
        self.context.transition_to(PreActionState())

class FinishedState(State):
    def run(self):
        pass

def main(question: str):
    logging.info("Starting")
    store = EmbeddingStore()
    context = Context(CreatePlanState(), question, store)
    while type(context._state) != FinishedState:
        context.run()
    print(context.answer)
    logging.info("Finished")


if __name__ == '__main__':
    # command = "Answer the question: What is the capital of France?"
    # command = "Answer the question: What are the official languages of Belgium, Singapore, and South Africa?"
    # command = "Answer the question: What is the height of the tallest building in New York City and who is the architect?"
    command = "Answer the question: What is the combined GDP of China, Japan, US, Cananda, and UK in 2022?"
    main(command)
