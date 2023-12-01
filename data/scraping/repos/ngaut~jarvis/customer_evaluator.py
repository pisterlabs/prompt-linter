import re
import yaml
import openai
from typing import Optional, Any, Dict

from langsmith import RunEvaluator
from langchain.evaluation.schema import StringEvaluator
from langsmith.evaluation import EvaluationResult

from langchain.chains import LLMChain

from jarvis.smartgpt import gpt
from jarvis.smartgpt import instruction


class GrammarAccuracyEvaluator(StringEvaluator):
    def _check_jvm_syntax(self, prediction: str) -> bool:
        # Checking for valid jvm.eval and jvm.get calls
        eval_pattern = re.compile(r"jvm\.eval\([^)]+\)")
        get_pattern = re.compile(r"jvm\.get\([^)]+\)")

        if not all(
            [
                bool(re.fullmatch(eval_pattern, s))
                for s in re.findall(r"jvm\.eval\([^)]+\)", prediction)
            ]
        ):
            return False
        if not all(
            [
                bool(re.fullmatch(get_pattern, s))
                for s in re.findall(r"jvm\.get\([^)]+\)", prediction)
            ]
        ):
            return False

        # Add more checks for other syntax rules if needed
        return True

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self._check_jvm_syntax(prediction):
            return {
                "score": 1.0,
                "value": prediction,
                "reasoning": "The prediction follows the correct JVM syntax.",
            }
        else:
            return {
                "score": 0.0,
                "value": prediction,
                "reasoning": "The prediction does not follow the correct JVM syntax.",
            }


class YAMLCorrectnessEvaluator(StringEvaluator):
    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            yaml.safe_load(prediction)
            score = 1.0
            reasoning = "The prediction is a valid YAML format."
        except yaml.YAMLError as e:
            score = 0.0
            reasoning = str(e)
        return {
            "score": score,
            "value": prediction,
            "reasoning": reasoning,
        }


class InstructionValidityEvaluator(RunEvaluator):
    LLM_TEMPLATE = """
    Here are a set of instructions:
    --------
    {instructions}
    --------
    On a scale from 0 to 100, how valid and complete do these instructions appear? Provide a score at the end.
    """

    def __init__(self):
        self.llm = self._initialize_llm()

    @staticmethod
    def _initialize_llm():
        """Initialize the ChatOpenAI based on the API type."""
        return gpt.OPEN_AI_MODELS_HUB[gpt.GPT_4].get_llm()

    def evaluate_run(self, run, example: Optional[dict] = None) -> EvaluationResult:
        output_string = run.outputs.get("output")
        output_content = yaml.safe_load(output_string)
        task = output_content.get("task")

        instructions_list = self._get_instructions_from_output(output_content)
        if not instructions_list:
            return EvaluationResult(
                key="InstructionValidity", score=0, details="No instructions found"
            )

        return self._execute_and_evaluate_instructions(instructions_list, task)

    @staticmethod
    def _get_instructions_from_output(output_content):
        """Extract instructions if they exist and are non-empty."""
        instructions = output_content.get("instructions")
        return instructions if isinstance(instructions, list) and instructions else None

    def _execute_and_evaluate_instructions(self, instructions, task):
        try:
            interpreter = instruction.JVMInterpreter()
            interpreter.run(instructions, task=task)
            execution_result = interpreter.get_results()
        except Exception as e:
            return EvaluationResult(key="InstructionValidity", score=0, details=str(e))

        return self._evaluate_execution_result(execution_result)

    def _evaluate_execution_result(self, execution_result):
        evaluator_result = LLMChain.from_string(
            llm=self.llm, template=self.LLM_TEMPLATE
        )(dict(instructions=execution_result))
        score = self._extract_score_from_evaluator_result(evaluator_result)
        return EvaluationResult(key="InstructionValidity", score=score)

    @staticmethod
    def _extract_score_from_evaluator_result(evaluator_result):
        """Extract score from the AI's evaluator result."""
        score_match = re.search(r"\d+", evaluator_result["text"])
        return float(score_match.group(0).strip()) / 100.0 if score_match else 0
