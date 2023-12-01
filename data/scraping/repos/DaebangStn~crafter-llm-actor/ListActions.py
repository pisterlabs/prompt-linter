import json
from logging import Logger
from typing import List, Dict, Optional, Deque

from langchain import PromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.schema import BaseRetriever

from agent import serializer_surrounding_deque, serializer_inventory_deque


class ListActions:
    llm: BaseLanguageModel
    chain: LLMChain
    logger: Optional[Logger]

    prompt: str = """I will give you some information about strategical open world game.
    List top 3 actions the player should take to complete the task and the requirement for each action. 
    Explain how your action benefits the given task. Say what's most urgent first
    Imagine that you are an expert player who thinks strategically. Choose ONLY from the list of all actions.   

    You must follow: If you can't determine the correct answer with the information you have, select "Noop".

    {context}

    Available Actions: 
    {actions}

    Top 3 Actions, Explain and Requirements:"""

    def __init__(self, llm: BaseLanguageModel, logger: Logger, _retriever: BaseRetriever):
        self.llm = llm
        self.logger = logger

        assert llm is not None, "Must have a language model"

        _prompt = PromptTemplate(
            template=self.prompt, input_variables=["context", "actions"]
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=_prompt,
        )

        # chain_type_kwargs: dict = {"prompt": _prompt}
        # self.chain = RetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=_retriever,
        #     return_source_documents=True,
        #     chain_type_kwargs=chain_type_kwargs,
        #     verbose=True,
        # )

    def run(
            self,
            _available_actions: List[Dict[str, str]],
            _surrounding_deque: Optional[Deque[Dict[str, list]]] = None,
            _inventory_deque: Optional[Deque[Dict[str, int]]] = None,
            **kwargs: Optional[str]
    ) -> str:
        assert len(_available_actions) > 0, "Must have at least one action"
        _actions = json.dumps(_available_actions, indent=1)

        status_explain = []
        if _surrounding_deque:
            status_explain.append(serializer_surrounding_deque(_surrounding_deque))
        if _inventory_deque:
            status_explain.append(serializer_inventory_deque(_inventory_deque))

        for key, value in kwargs.items():
            status_explain.append(f"{key}:\n{value}")

        _context = "\n".join(status_explain)
        context_logger = _context.replace('\n', ' ')
        self.logger.info(f"Context: {context_logger}")
        response = self.chain.run(context=_context, actions=_actions)
        response_logger = response.replace('\n', ' ')
        self.logger.info(f"Response: {response_logger}")

        return response
