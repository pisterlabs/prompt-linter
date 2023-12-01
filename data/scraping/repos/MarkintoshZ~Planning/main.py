import os
import logging
from textwrap import dedent
from dotenv import load_dotenv
import pandas as pd
import guidance
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from duckduckgo_search import DDGS
from utils import fetch_html, flatten, get_chunks
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
    level=logging.INFO,
)


class EmbeddingStore:
    def __init__(self, model="text-embedding-ada-002"):
        self.store = pd.DataFrame(columns=["text", "embedding"])
        self.model = model

    def add(self, text):
        embedding = get_embedding(text, engine=self.model)
        self.store[len(self.store.index)] = [text, embedding]

    def search(self, text, n=3):
        embedding = get_embedding(text, engine=self.model)
        self.store["similarities"] = self.store["embedding"].apply(
            lambda x: cosine_similarity(x, embedding)
        )
        res = self.store.sort_values("similarities", ascending=False).head(n)["text"]
        return res


def search(query: str):
    return DDGS().text(query)


def format_search_result(result):
    return f"# {result['title']}\n{result['href']}\n{result['body']}\n"


class CreatePlanState(State):
    def run(self):
        program = guidance(
            dedent(
                """\
            {{#system~}}
            You are a helpful agent
            {{~/system}}
            {{#user~}}
            Create a step-by-step search engine query plan that fulfills the given command by user.

            Command: Answer the question: Who wrote the play 'Hamlet,' and where was he born?
            Plan:
            1. find the author of 'Hamlet' and edit the answer
            2. find where the author was born in and edit the answer

            Command: Answer the question: Who are the current Prime Ministers of the UK and Canada, and who has a higher educational background?
            Plan:
            1. find the current Prime Minister of UK and edit the answer
            2. find the Prime Minister of Canada and edit the answer
            3. compare the educational background of the two Prime Ministers and edit the answer

            Command: {{command}}
            Plan:
            {{~/user}}
            {{#assistant~}}
            {{gen 'plan' temperature=0.0 max_tokens=200}}
            {{~/assistant}}"""
            )
        )
        results = program(command=self.context.command, llm=LLM)
        logging.info(results)
        self.context.plan = results["plan"]
        self.context.transition_to(PreActionState())


class PreActionState(State):
    def run(self):
        program = guidance(
            dedent(
                """\
            {{#system~}}
            You are a helpful agent
            {{~/system}}
            {{#user~}}
            User command: {{command}}
            Current answer: {{current_answer}}
            Plan: {{plan}}
            Previous Actions: {{previous_actions}}
            Location: {{location}}
            Content: {{content}}
            Available Actions:
            - web-search [keywords] (search keywords using a search engine)
            - semantic-memory-search [text] (search for semantically similar text segments at the current location)
            - edit-answer [new-answer] (edit and update the current answer. Apply this action if the last action is web-search)
            - navigate [url] (navigate to the given url)
            - replan [plan] (update the current plan. Each step of the plan should be on its own line)
            - terminate (terminate when the answer is complete)
            Choose an action by outputing the action name followed by the action arguments.
            {{~/user}}
            {{#assistant~}}
            {{gen 'action' temperature=0.1 max_tokens=150}}
            {{~/assistant}}"""
            )
        )
        results = program(
            command=self.context.command,
            current_answer=self.context.answer,
            plan=self.context.plan,
            previous_actions=", ".join(
                [f"{i+1}. {s}" for i, s in enumerate(self.context.actions)]
            ),
            location=self.context.location,
            content=self.context.content,
            llm=LLM,
        )
        logging.info(results)
        action: str = results["action"].strip()
        self.context.transition_to(ExecuteActionState(action))


class ExecuteActionState(State):
    def __init__(self, action) -> None:
        super().__init__()
        self.action = action

    def run(self):
        # parse and validate action
        action = self.action
        if action.find("replan") == 0:
            self.context.actions.append("replan")
        else:
            self.context.actions.append(action)
        # print action in red
        print(f"\033[91mAction: {action}\033[00m")
        if action.find("web-search") != -1:
            query = action.split("web-search")[1].strip()
            results = list(search(query))
            print("Result len: ", len(results))
            results = [format_search_result(res) for res in results]
            # for result in results[:5]:
            #     self.context.embedding_store.add(result)
            self.context.localtion = "Duckduckgo Search Result"
            self.context.content = "\n".join(results[:5])
            self.context.transition_to(PreActionState())
        elif action.find("navigate") != -1:
            url = action.split("navigate")[1].strip()
            self.context.buffer = flatten(fetch_html(url))
            self.context.location = url
            self.context.content = (
                "Page loaded. Use semantic-memory-search to search for text segments."
            )
            self.context.transition_to(PreActionState())
        elif action.find("semantic-memory-search") != -1:
            query = action.split("semantic-memory-search")[1].strip()
            if self.context.content == "":
                self.context.content = "No results found."
                self.context.transition_to(PreActionState())
            store = EmbeddingStore()
            for chunk in get_chunks(self.context.buffer, 500, 75):
                store.add(chunk)
            search_results = store.search(query, n=3)
            if len(search_results) == 0:
                self.context.content = "No results found."
                self.context.transition_to(PreActionState())
            else:
                result_strs = [str(result) for result in search_results]
                print(result_strs)
                self.context.content = str("\n---\n".join(result_strs))
                self.context.transition_to(PreActionState())
        elif action.find("edit-answer") != -1:
            answer = action.split("edit-answer")[1].strip()
            self.context.answer = answer
            self.context.transition_to(PreActionState())
        elif action.find("replan") != -1:
            self.context.plan = action.split("replan")[1].strip()
            self.context.transition_to(PreActionState())
        elif action.find("terminate") != -1:
            self.context.transition_to(FinishedState())
        else:
            self.context.transition_to(PreActionState())


class ReplanState(State):
    def __init__(self, action_results):
        super().__init__()
        self.action_results = action_results

    def run(self):
        program = guidance(
            dedent(
                """\
            {{#system~}}
            You are a helpful agent
            {{~/system}}
            {{#user~}}
            User command: {{command}}
            Current answer: {{current_answer}}
            Plan: {{plan}}
            Action chosen: {{action}}
            Results: {{action_results}}
            Create an updated plan. Include in the plan the answer to compose for the next step if necessary. You do not need to "confirm" anything. Terminate if the answer is complete, even if plan is not completed. Make sure not to deviate from the original command.
            {{~/user}}
            {{#assistant~}}
            {{gen 'new_plan' temperature=0.1 max_tokens=150}}
            {{~/assistant}}"""
            )
        )
        results = program(
            command=self.context.command,
            plan=self.context.plan,
            current_answer=self.context.answer,
            action=self.context.actions[-1],
            action_results=self.action_results,
            llm=LLM,
        )
        logging.info(results)
        self.context.plan = results["new_plan"].strip()
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


if __name__ == "__main__":
    # command = "Answer the question: What is the capital of France?"
    # command = "Answer the question: What are the official languages of Belgium, Singapore, and South Africa?"
    # command = "Answer the question: What is the height of the tallest building in New York City and who is the architect?"
    command = "Answer the question: What is the combined GDP of China, Japan, US, Cananda, and UK?"
    main(command)
