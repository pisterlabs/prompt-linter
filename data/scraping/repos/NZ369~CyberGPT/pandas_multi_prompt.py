from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain

from langchain.chains.router.base import MultiRouteChain, RouterChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain.agents.agent import AgentExecutor
from pandas import DataFrame

class PandasMultiPromptChain(MultiRouteChain):
    """
    CC: Modification of MultiPromptChain for switching prompts on a csv agent.
    Destination chains are AgentExecutors (chains that execute agents),
    instead of LLM Chains directly.  The default chain is also an AgentExecutor
    instead of an LLM Chain.
    """

    router_chain: RouterChain
    """Chain for deciding a destination chain and the input to it."""
    destination_chains: Mapping[str, AgentExecutor]
    """Map of name to candidate chains that inputs can be routed to."""
    default_chain: AgentExecutor 
    """Default chain to use when router doesn't map input to one of the destinations."""
    output_key: str = "output"  #: :meta private:
    
    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    @classmethod
    def from_prompts(
        cls,
        llm: BaseLanguageModel,
        prompt_infos: List[Dict[str, Any]],
        default_chain: Optional[LLMChain] = None,
        **kwargs: Any,
    ) -> PandasMultiPromptChain:

        """Convenience constructor for instantiating from destination prompts."""

        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations_str
        )
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            chain = p_info["agent_chain"]
            destination_chains[name] = chain
        _default_chain = default_chain or ConversationChain(llm=llm, output_key="text")

        return cls(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=_default_chain,
            **kwargs,
        )