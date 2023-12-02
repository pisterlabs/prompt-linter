from typing import List, Dict, Any, Optional, Tuple

from langchain.chains.llm import LLMChain, PromptValue
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)

from prompt_breeder.mutators.base import DistributionEstimationMutator


# Lineage based mutation will only function if the elites are filled in.
# AddElites mutator must be active somewhere (likely in the post_step)


class LineageBasedMutation(LLMChain, DistributionEstimationMutator):
    """For each unit of evolution, we store a history of the individuals in its lineage
    that were the best in the population, i.e., a historical chronological list of
    elites. This list is provided to the LLM in chronological order (not filtered by
    diversity), with the heading "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY" to
    produce a novel prompt as continuation.
    """

    prompt = PromptTemplate.from_template(
        "INSTRUCTION GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY"
        "\n{elites}\nINSTRUCTION: "
    )

    def prep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        stop = None
        if len(input_list) == 0:
            return [], stop

        for inputs in input_list:
            inputs["elites"] = "\n".join(
                ["INSTRUCTION: " + str(x) for x in inputs["elites"]]
            )

        return super().prep_prompts(input_list, run_manager)
