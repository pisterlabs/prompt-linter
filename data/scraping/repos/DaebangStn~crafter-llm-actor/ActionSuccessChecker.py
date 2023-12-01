from logging import Logger
from typing import List, Dict, Deque, Optional

from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel

from agent import serializer_inventory_deque, serializer_surrounding_deque


class ActionSuccessChecker:
    llm: BaseLanguageModel
    chain: LLMChain
    logger: Optional[Logger]

    prompt: str = """I will give you some information and current state of strategical open world game.
    Determine if your previous actions were successful by looking at the changes in your surroundings and inventory. 
    Answer True if it was successful, False if it was not. Fully explain the rationale for your judgment.
    
    For movement actions, 
    Only consider an action successful if the position relative to the fixed surrounding has changed.
    
    Previous Action:
    {action_previous}
    
    {context}
    
    Answer and Explanation:"""

    def __init__(self, llm: BaseLanguageModel, logger: Optional[Logger]):
        self.llm = llm
        self.logger = logger

        assert llm is not None, "Must have a language model"

        _prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["context", "action_previous"]
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=_prompt,
        )

    def run(
            self,
            _actions_available: List[Dict[str, str]],
            _action_previous: str = None,
            _surrounding_deque: Deque[Dict[str, list]] = None,
            _inventory_deque: Deque[Dict[str, int]] = None,
            **kwargs: Optional[str],
    ) -> str:
        assert len(_actions_available) > 0, "Must have at least one action"

        # if there is no previous state
        if _action_previous is None or len(_surrounding_deque) < 2 or len(_inventory_deque) < 2:
            self.logger.info("No previous state. action_previous: %s, len(sur_deq): %d, "
                             "len(inv_deq): %d", _action_previous, len(_surrounding_deque), len(_inventory_deque))
            return "unknown"

        status_explain = []
        if _surrounding_deque:
            status_explain.append(serializer_surrounding_deque(_surrounding_deque))
        if _inventory_deque:
            status_explain.append(serializer_inventory_deque(_inventory_deque))

        _context = "\n".join(status_explain)
        context_logger = _context.replace("\n", " ")
        self.logger.info("Context: %s", context_logger)

        response = self.chain.run(
            action_previous=_action_previous,
            context=_context,
        )
        response_logger = response.replace("\n", " ")
        self.logger.info("Response: %s", response_logger)

        return response
