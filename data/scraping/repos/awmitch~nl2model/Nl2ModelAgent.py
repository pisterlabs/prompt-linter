import re

from langchain.schema import AgentAction, AgentFinish
from langchain.agents import LLMSingleActionAgent

from typing import List, Tuple, Any

# Custom Agent
class Nl2ModelAgent(LLMSingleActionAgent):
    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."}, ""
            )
        elif early_stopping_method == "generate":
            # Generate does one final forward pass
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += (
                    f"\nObservation:{observation}\n"
                )
            # Adding to the previous steps, we now tell the LLM to make a final pred
            thoughts += (
                "\n\nI now need to return a final response based on the previous steps.\nFinal Response:"
            )
            new_inputs = {"agent_scratchpad": thoughts, "intermediate_steps":intermediate_steps}
            full_inputs = {**kwargs, **new_inputs}
            full_output = self.llm_chain.run(**full_inputs)
            # We try to extract a final answer
            return AgentFinish(return_values={"output": full_output.split("Final Response:")[-1].strip()},
                log=full_output,
                )
        else:
            raise ValueError(
                "early_stopping_method should be one of `force` or `generate`, "
                f"got {early_stopping_method}"
            )