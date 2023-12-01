import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, Wikipedia
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX
from langchain.agents.react.base import DocstoreExplorer
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chat_models import ChatOpenAI
from langchain.load.dump import dumps
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.utils.input import get_color_mapping

from prompts import CONTEXT_SUFFIX, NO_CONTEXT_SUFFIX
from utils import normalize_answer

load_dotenv(find_dotenv())

class CustomAgent(ZeroShotAgent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def build_retro_prompt(self,
                         question: str,
                         context: str,
                         intermediate_steps: List[Tuple[AgentAction: str]],
                         success: bool
                         ) -> str:
    """ 
      Builds a retroformer prompt from the agent's intermediate steps
    """
    prompt = ""
    prompt += f"Question: {question}\n\n"

    if len(context) > 0:
      prompt += f"Context: {context}\n\n"
    
    prompt += f"{self.llm_prefix}"

    for action, observation in intermediate_steps:
          prompt += action.log
          if observation is not None:
            prompt += f"\n{self.observation_prefix}{observation}\n\n{self.llm_prefix}"

    if success:
      prompt += f"\nThe final answer was CORRECT"
    else:
      prompt += f"\nThe final answer was INCORRECT"
     
    return prompt

class CustomExecutor(AgentExecutor):
  """ Modified agent executor that saves the final (finish task) reasoning step as an intermediate step."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )

            if isinstance(next_step_output, AgentFinish):
                next_step_intermediate = AgentAction(
                   "AgentFinish", "None", next_step_output.log
                )
                intermediate_steps.extend([(next_step_intermediate, None)]) # Add the last reasoning log to the intermediate steps
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

class Actor:
  def __init__(self,
               task_id: int,
               task: Any,
               with_context: bool=True,
               model: str="gpt-4",
               model_temperature: float=0,
               f1_threshold: float=0.7):
    self.task_id = task_id
    self.task = task
    self.question = task["question"]
    self.with_context = with_context
    self.context = None
    if self.with_context:
      self.context = task['supporting_paragraphs']
    self.answer = task["answer"]
    print(f"task_id: {self.task_id}\n Question: {self.question}\n\n Supporting Paragraph: {self.context}\n\n Answer: {self.answer}\n\n")

    self.episode = 0
    self.actor_model = ChatOpenAI(model=model, temperature=model_temperature) # define the actor model (frozen weights) (probably some gpt-4)
    self.f1_threshold = f1_threshold # f1 score threshold that determines correct/incorrect
    self.current_policy = "" # Initialize with an empty policy prompt
    self.docstore = DocstoreExplorer(Wikipedia())
    self.tools = [
      Tool(
        name="Search",
        func=self.docstore.search,
        description="useful for when you need to ask with search"
      ),
      Tool(
        name="Lookup",
        func=self.docstore.lookup,
        description="(only use after a search) useful for when you need to ask with lookup"
      )
    ]

    if self.with_context:
      self.prompt = ZeroShotAgent.create_prompt(
        self.tools,
        prefix=PREFIX,
        format_instructions=FORMAT_INSTRUCTIONS,
        suffix=CONTEXT_SUFFIX,
        input_variables=["input", "agent_scratchpad", "context", "policy", "long_term_memory"]
      )
    else:
      self.prompt = ZeroShotAgent.create_prompt(
        self.tools,
        prefix=PREFIX,
        format_instructions=FORMAT_INSTRUCTIONS,
        suffix=NO_CONTEXT_SUFFIX,
        input_variables=["input", "agent_scratchpad", "policy", "long_term_memory"]
      )

    self.llm_chain = LLMChain(llm=self.actor_model, prompt=self.prompt)
    self.tool_names = [tool.name for tool in self.tools]
    self.agent = CustomAgent(llm_chain=self.llm_chain,
                             allowed_tools=self.tool_names,
                             max_iterations=10,
                             return_intermediate_steps=True)
    
    self.long_term_memory = [] # The reflection responses that summearize prior failed attemps
    
  def add_reflection_response(self, reflection):
    self.long_term_memory.append(reflection)
  
  def clear_reflection_response(self):
    self.long_term_memory =[]

  
  def f1_score(self, reference, candidate):
      """
      Calculate the F1 score between a reference answer and a candidate answer.
      
      Args:
      - reference (str): the reference (or gold standard) answer
      - candidate (str): the model-generated answer
      
      Returns:
      - float: the F1 score between the two answers
      """
      
      # Tokenize the answers into words
      reference_tokens = set(reference.split())
      candidate_tokens = set(candidate.split())
      
      # Calculate the number of shared tokens between the two answers
      common = reference_tokens.intersection(candidate_tokens)
      
      # If there are no shared tokens, the F1 score is 0
      if not common:
          return 0.0
  
      # Calculate precision and recall
      precision = len(common) / len(candidate_tokens)
      recall = len(common) / len(reference_tokens)
      
      # Calculate the F1 score
      f1 = 2 * (precision * recall) / (precision + recall)
      
      return f1

  def get_reward(self, reference_answer: str, candidate_answer: str) -> float:
    """ Calculate the f1 score for the agent answer"""
    return self.f1_score(normalize_answer(reference_answer), normalize_answer(candidate_answer))
  

  def format_longterm_memory(self) -> str:
    """ Generates a prompt string from the list of long term reflections from the agent
        [".1.", " .2. ", ".3."] ->
       output format:
        \n
        .1. \n
        .2. \n
        .3. \n
        \n
    """
    
    formatted_reflections = "\n".join([reflection.strip() for reflection in self.long_term_memory])
    return f"\n\n{formatted_reflections}\n\n"
  
  def update_policy(self, reflection: str):
    """ Update the policy prompt with the reflection response from the retroformer"""
    # Add the current policy to the long term memory
    self.long_term_memory.append(self.current_policy)

    # Update the current policy with the reflection response
    self.current_policy = reflection
  
  
  def _handle_error(self, error) -> str:
    return str(error)[:50]
    

  def rollout(self) -> Dict:
    """ Get a single agent answer under the current policy"""
    output = None
    agent_executor = CustomExecutor.from_agent_and_tools(
      agent=self.agent,
      tools=self.tools,
      verbose=True,
      handle_parsing_errors=self._handle_error,
      return_intermediate_steps=True,
    )

    try:
      response = agent_executor(
        {
          "input": self.question,
          "context": self.context,
          "policy": self.current_policy, # latest entry of the long term memory
          "long_term_memory": self.format_longterm_memory()
        }
      )
    except ValueError as error:
      print(error)
      response = {
        "input": self.question,
        "context": self.context,
        "policy": self.current_policy,
        "long_term_memory": self.format_longterm_memory(),
        "output": "No answer because lookup without a search.",
        "intermediate_steps": [(AgentAction(tool='Fail', tool_input='None', log="I Tried to do a lookupt before i did a search. I should not do this."), None)]
      }

    f1_score = self.get_reward(self.answer, response["output"])

    # bool that is true if f1 > self.threshold, false otherwise
    success = f1_score >= self.f1_threshold
    
    prompt = self.agent.build_retro_prompt(response["input"],
                                           response["context"],
                                           response["intermediate_steps"],
                                           success)
  
    self.episode += 1
    return {
      "task_id": self.task_id,
      "response": response["output"],
      "reflection_prompt": prompt,
      "f1_score": f1_score
    }

  def test(self):
    """ 
      Perform run under the current policy.
    """
    
    agent_executor = CustomExecutor.from_agent_and_tools(
      agent=self.agent,
      tools=self.tools,
      verbose=True,
      handle_parsing_errors=self._handle_error,
      return_intermediate_steps=True,
    )
    
    try: 
      response = agent_executor(
            {
              "input": self.question,
              "context": self.context,
              "policy": self.current_policy,
              "long_term_memory": self.format_longterm_memory()
            }
          )
    except ValueError as error:
      print(error)
    print("response", response)

    f1_score = self.get_reward(self.answer, response["output"])

    print("f1_score", f1_score)
    succcess = f1_score >= self.f1_threshold

    prompt = self.agent.build_retro_prompt(response["input"],
                                           response["context"], 
                                           response["intermediate_steps"],
                                           succcess
                                           )
    
    print("Prompt: \n\n")
    print(prompt)
    print("\n\n")



if __name__ == "__main__":
  task_id = 0
  task = {}
  task = {
    "question": "Which harry potter film series main stars debuted in stage acting first?",
    "answer": "Daniel Radcliffe",
    "supporting_paragraphs": "harry potter is a movie about a guy with a scar on his head"
  }

  actor = Actor(task_id, task)
  actor.test()
  