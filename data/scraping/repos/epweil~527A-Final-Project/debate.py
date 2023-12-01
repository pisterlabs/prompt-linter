##Ethan Weilheimer 
##CSE 527A LLM Agent Paper 
##Code adapted from Cobus Greyling (https://cobusgreyling.medium.com/two-llm-based-autonomous-agents-debate-each-other-e13e0a54429b)
import time
from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field
from langchain.tools import tool
from utils import read_json_file, VIEW_DEBATE
from langchain.llms import VertexAI


class DialogueAgent:
    def __init__(self, name: str, system_message: SystemMessage, model: ChatOpenAI, stop: List[str]) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.stop = stop

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        message = self.model(self.system_message + '\n' + "\n".join(self.message_history + [self.prefix]))
        return message

    def receive(self, name: str, message: str) -> None:
        if name is None:
            self.message_history.append(message)
        else:
            self.message_history.append(f"{name}: {message}")


class DialogueSimulator:

    def __init__(self, agents: List[DialogueAgent], moderator_name: str, moderator_message: str,
                 selection_function: Callable[[int, List[DialogueAgent]], int]) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function
        self.moderator_name = moderator_name
        self.moderator_message = moderator_message

    def void_step(self):
        self._step += 1

    def reset(self):
        for agent in self.agents:
            agent.reset()
            agent.receive(self.moderator_name, self.moderator_message)

    def step(self) -> tuple[str, str]:
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
        message = speaker.send()

        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        self._step += 1

        return speaker.name, message


def generate_system_message(name, adj):
    return f"""

Your name is {name}.

Your purpose is as follows: A human will present you with a "Problem" followed by their "Proposed Solution". You will also be provided with the human's "Previous Actions" that they have took. Your goal is to provide an argument for why "Proposed Solution" is {adj} given the "Problem". Try to be as convincing as possible, as you will be debating another agent. You should speak directly to the human, you are trying to convince them that their "Proposed Solution" is {adj}.

If applicable, you should directly address the argument made by the other agent to show why it is not a strong argument. Provide rebuttals to their argument, or additional claims to your own. Additionally, in your arguments you should take into consideration the past actions of the user.



DO NOT use more than 3-5 sentences of text.
DO NOT fabricate fake citations or claims.
DO NOT restate the Problem or Proposed Solution. Get straight to the point.
DO NOT add anything else.
Stop speaking the moment you finish giving your argument.
"""


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx


class ViewDebate(BaseModel):
    problem_and_proposed_solution: str = Field(description="A description of the problem or goal you are trying to solve followed by your proposed solution to this problem or goal.")


def view_debate_wrapper(context, total_iters=2, temperature=0, negative_first=False, model='text-bison-32k', model_type='text'):
    @tool(VIEW_DEBATE, args_schema=ViewDebate)
    def view_debate(problem_and_proposed_solution):
        """Use this tool to view a debate on whether your action is the best or not. You should use this tool to get a better understanding about the best solution to your problem. You will receive a dialogue between 2 debaters who are arguing whether your proposed action is best or not."""

        previous_actions = '\n'.join(context.generation_observation_history)
        # return "Your action is not the best action."
        situation = f"Previous Actions:{previous_actions}\nSituation: {problem_and_proposed_solution}"

        # print('IN DEBATE:\n' + situation + '\nEND SITUATION')

        names_and_info = {
            "AI affirm": 'the best possible solution',
            "AI negative": 'NOT the best possible solution (i.e., there exists a better solution)',
        }

        agent_system_messages = {
            name: generate_system_message(name, names_and_info[name])
            for name in names_and_info
        }

        stop = ['\n']


        if model_type == 'text':
            affirm_llm = VertexAI(model_name=model, temperature=temperature)
            negative_llm = VertexAI(model_name=model, temperature=temperature)
        # TODO: handle the chat model types
        else:
            affirm_llm = None
            negative_llm = None

        agents = [
            DialogueAgent(
                name="AI affirm",
                system_message=agent_system_messages["AI affirm"],
                model=affirm_llm,
                stop=stop
            ),
            DialogueAgent(
                name="AI negative",
                system_message=agent_system_messages["AI negative"],
                model=negative_llm,
                stop=stop
            )
        ]

        simulator = DialogueSimulator(
            agents=agents,
            moderator_message=situation,
            selection_function=select_next_speaker,
            moderator_name='',
        )
        simulator.reset()

        if negative_first:
            simulator.void_step()

        debate_history = []
        for _ in range(total_iters):
            # time.sleep(5)
            print(_)
            name, message = simulator.step()
            debate_history.append(f"{name}: {message}".strip())

        return "\n".join(debate_history)

    return view_debate

