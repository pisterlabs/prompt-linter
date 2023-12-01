from re import search
from typing import Any, Optional
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation import StringEvaluator

class RelevanceEvaluator(StringEvaluator):
    def __init__(self):
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        # self.sistem_prompt = "Write only in the Serbian language."
        prompt = """On a scale from 1 to 5, how correct is the following response to the input:
        -------- INPUT: {input}
        -------- OUTPUT: {prediction}
        -------- While evaluating the response act as if you were a mathematician.
        -------- Check if the provided information is true or not.
        -------- Reason step by step about why the score is appropriate, then print the score at the end. At the end, repeat that score alone on a new line."""
        self.eval_chain = LLMChain.from_string(llm=llm, template=prompt)

    @property
    def requires_input(self) -> bool:
        return True

    @property
    def requires_reference(self) -> bool:
        return False

    @property
    def evaluation_name(self) -> str:
        return "scored_relevance"

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        evaluator_result = self.eval_chain(
            dict(input=input, prediction=prediction),
            **kwargs
        )
        reasoning, score = evaluator_result["text"].split("\n", maxsplit=1)
        score = search(r"\d+", score).group(0)
        if score is not None:
            score = float(score.strip())
        return {"score": score, "reasoning": reasoning.strip()}
