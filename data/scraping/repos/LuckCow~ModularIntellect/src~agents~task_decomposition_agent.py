from langchain import LLMChain, PromptTemplate
from langchain.callbacks import CallbackManager, StdOutCallbackHandler
from langchain.llms import BaseLLM

from src.components.itask import ITask
from src.services.chainlang_agent_service import BaseChainLangAgent


class TaskDecompositionAgent(BaseChainLangAgent):
    """Breaks a complex task into subtasks that can be assigned to another agent"""

    # Task Creation Prompt
    PROMPT = """Please systematically break down this objective into a list of subtasks that can be completed by a single 
specialized agent. Each agent can solve a particular type of problem as specified in its description. Each task consists of a 
task_description and a task_justification. The task_description is a short description of the task that explains what
the agent should do to complete the task. The task_justification explains how completing the task will contribute to
the overall objective.
The output should be a list of tasks in the format: [task_number]. [assigned_agent] | [task_description] | [task_justification]
Specialized Agents (Name - Description): 
{agent_list}
{past_tasks}
The objective is: {objective}
Subtasks to Complete:"""

    # Agent List: list of agents that can complete tasks and descriptions of how they accomplish it
    AGENT_LIST = """RESEARCHER - Given a topic or question, this agent will search the internet for relevant information and store it for future reference.
CALCULATOR - Given a math question, this agent will calculate the exact result.
PROGRAMMER - Given a programming task, this agent will write the code to complete it.
WRITER - Given a general and simple task, this agent will write out the solution to it using results from previous tasks and memory.
Special Case Agents:
TASK_DECOMPOSER - Given a task that is too complicated for a single agent to complete, this agent will break it down into simpler subtasks that can be processed.
IMPOSSIBLE_TASK - Given a task that cannot be solved by any of the other agents, assign this task to this agent and it will explain why it is impossible."""

    # TODO: maybe add example for few shot learning

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt = PromptTemplate(input_variable=["objective", "agent_list", "past_tasks"], template=self.PROMPT)
        super().__init__()

    def _get_chain(self):
        """Create a ChatVectorDBChain for question/answering."""
        # Construct a ChatVectorDBChain with a streaming llm for combine docs
        # and a separate, non-streaming llm for question generation

        manager = CallbackManager([])
        manager.set_handler(StdOutCallbackHandler())

        question_generator = LLMChain(
            llm=self.llm, prompt=self.prompt, callback_manager=manager, verbose=True
        )

        return question_generator

    def execute(self, task: ITask):
        self._chain.predict({"objective": task, "agent_list": self.AGENT_LIST, "past_tasks": ""})



# Example:
"""Please systematically break down this objective into a list of subtasks that can be completed by a single 
specialized agent. Each agent can solve a particular type of problem as specified in its description. Each task consists of a 
task_description and a task_justification. The task_description is a short description of the task that explains what
the agent should do to complete the task. The task_justification explains how completing the task will contribute to
the overall objective. Note that there are 2 special case agents for tasks that do not fit into the other categories.
The output should be a list of tasks in the format: [task_number]. [assigned_agent] | [task_description] | [task_justification]
Task Agents (Name - Description): 
RESEARCHER - Given a topic or question, this agent will search the internet for relevant information and store it for future reference.
CALCULATOR - Given a math question, this agent will calculate the exact result.
PROGRAMMER - Given a programming task, this agent will write the code to complete it.
WRITER - Given a general and simple task, this agent will write out the solution to it using results from previous tasks and memory.
Special Case Agents:
TASK_DECOMPOSER - Given a task that is too complicated for a single agent to complete, this agent will break it down into simpler subtasks that can be processed.
IMPOSSIBLE_TASK - Given a task that cannot be solved by any of the other agents, assign this task to this agent and it will explain why it is impossible.
The objective is: Find the weight and diameter of the planet Jupiter in terms of the weight and length of a 2003 Honda Accord.
Justification of Objective: I want to give a relatable idea of scale for a presentation I am giving.
Subtasks to Complete:"""

# Result:
"""    RESEARCHER | Find the weight and dimensions of a 2003 Honda Accord | This information will serve as a reference for comparison with Jupiter's weight and diameter.
    RESEARCHER | Find the weight and diameter of the planet Jupiter | This information will be used to compare with the weight and dimensions of a 2003 Honda Accord.
    CALCULATOR | Convert Jupiter's weight and diameter into terms of 2003 Honda Accord's weight and length | This will allow for a relatable comparison of Jupiter's weight and diameter.
    WRITER | Write a concise explanation of the comparison between Jupiter's weight and diameter and the weight and length of a 2003 Honda Accord | This explanation will help the user convey a relatable idea of scale for their presentation."""


# multistep prompt test
"""Answer the following questions as best you can, show your work throughout a step by step process. 

Use the following format:

Question: the input question you must answer
Thought: list out all of the possible next steps to take
Action: choose the action that will most quickly lead to the goal
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: The problem is now solved
Final Answer: the final state has been reached and the question is answered

Begin! Remember to take as many steps as needed to get to the final answer.

Question: {input}
If you are given a 5 gallon bucket and a 3 gallon bucket, both with no volume markings, and an infinite supply of water, how would you get exactly 2 gallons in one of the buckets?
"""