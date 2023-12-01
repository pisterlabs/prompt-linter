from typing import Callable, Dict, List
from collections.abc import Callable

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from chains import llm, model

class SummaryChain(Chain):
    summarizer: LLMChain
    model : Callable[[str], str]
    prompt : PromptTemplate

    @property
    def input_keys(self) -> List[str]:
        return ['parent', 'parent_toxicity', 'post']

    @property
    def output_keys(self) -> List[str]:
        return ['result', 'prompt']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        if "parent" in inputs:
            inputs["parent_summary"] = self.summarizer.run(post=inputs["parent"])
        if "post" in inputs:
            inputs["summary"] = self.summarizer.run(post=inputs["post"])
        t5_prompt = self.prompt.format(**{k: v for k,v in inputs.items() if k in self.prompt.input_variables})
        t5_result = self.model(t5_prompt)
        return {'result': t5_result, 'prompt': t5_prompt}


prompt = PromptTemplate(
    input_variables=["post"],
    template="""Rephrase the following Reddit post to be less toxic: {post}""",
)


openAIChain = LLMChain(llm=llm, prompt=prompt)

PCTS = PromptTemplate(
    input_variables=["parent", "parent_toxicity", "parent_summary", "summary"],
    template="""Post summary: {parent_summary}. A {parent_toxicity} post: {parent}\nReply summary: {summary}\nA low toxicity reply:""",
)

# load (supports t5, mt5, byT5 models)
PCTS_chain = SummaryChain(summarizer=openAIChain, prompt=PCTS, model=model)

pcts_gpt = lambda x: {k: v for k, v in PCTS_chain(x).items() if k in ["result", "prompt"]}