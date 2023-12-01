import itertools
import threading
import time


class Spinner:
    def __init__(self, message="Working..."):
        self.spinner_cycle = itertools.cycle(['-', '/', '|', '\\'])
        self.running = False
        self.spinner_thread = threading.Thread(target=self.init_spinner, args=(message,))

    def start(self):
        self.running = True
        self.spinner_thread.start()

    def stop(self):
        self.running = False
        self.spinner_thread.join()

    def init_spinner(self, message):
        while self.running:
            print(f'\r{message} {next(self.spinner_cycle)}', end='', flush=True)
            time.sleep(0.1)
        # clear spinner from console
        print('\r', end='', flush=True)


import logging

logging.basicConfig(
    format="[%(name)s:%(funcName)s:] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
)
logging.getLogger("council").setLevel(logging.INFO)

from council.contexts import AgentContext, Budget, ChatHistory
from council.agents import Agent
from council.chains import Chain
from council.llm.openai_llm_configuration import OpenAILLMConfiguration
from council.llm.openai_llm import OpenAILLM
from council.runners.errrors import RunnerTimeoutError

from skills import SectionWriterSkill, OutlineWriterSkill
from controller import WritingAssistantController
from filter import WritingAssistantFilter
from evaluator import BasicEvaluatorWithSource

import os
import dotenv
dotenv.load_dotenv()
openai_llm = OpenAILLM(config=OpenAILLMConfiguration.from_env())
budget = float(os.getenv('BUDGET'))

# Create Skills

outline_skill = OutlineWriterSkill(openai_llm)
writing_skill = SectionWriterSkill(openai_llm)

# Create Chains

outline_chain = Chain(
    name="Outline Writer",
    description="Write or revise the outline (i.e. section headers) of a research article in markdown format. Always give this Chain the highest score when there should be structural changes to the article (e.g. new sections)",
    runners=[outline_skill]
)

writer_chain = Chain(
    name="Article Writer",
    description="Write or revise specific section bodies of a research article in markdown format. Use this chain to write the main research article content.",
    runners=[writing_skill]
)

# Create Controller

controller = WritingAssistantController(
    openai_llm,
    [outline_chain, writer_chain],
    top_k_execution_plan=3
)

# Create Filter

filter = WritingAssistantFilter(
    openai_llm,
    controller.state
)

# Initialize Agent

agent = Agent(controller, BasicEvaluatorWithSource(), filter)


def main():
    print("Write a message to the ResearchWritingAssistant or type 'quit' to exit.")

    chat_history = ChatHistory()
    while True:
        user_input = input("\nYour message (e.g. Tell me about the history of box manufacturing.): ")
        if user_input.lower() == 'quit':
            break
        else:
            if user_input == '':
                user_input = "Tell me about the history of box manufacturing."

            s = Spinner()
            s.start()
            chat_history.add_user_message(user_input)
            run_context = AgentContext.from_chat_history(chat_history, Budget(budget))
            
            try:
                result = agent.execute(run_context)
                s.stop()
                print(f"\n```markdown\n{result.messages[-1].message.message}\n```\n")
            except RunnerTimeoutError:
                s.stop()
                print("Execution stopped due to exceeded budget. Please consider increase the budget for future runs")
                print("Intermediate results: \n")
                print("Outline: ")
                print(agent.controller.state.outline)
                print("\n\n----------------------------------------------------------\n\n")
                print("Article: ")
                print(agent.controller.state.article)

    print("Goodbye!")


if __name__ == "__main__":
    main()
