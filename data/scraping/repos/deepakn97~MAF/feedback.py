import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import func_timeout
from typing import Dict, List
from src.utils import OSFeedback, Prompt
import pandas as pd
from langchain.python import PythonREPL
from src.utils import OSFeedback, LLMFeedback, FeedbackFactory, Feedback
from src.gsm_maf.task_iterate import GSMIterate

# import torch
# importing func_timeout and torch causes segmentation fault on my machine


@FeedbackFactory.register("syntax")
class PythonExecuter(Feedback):
    def __init__(
        self,
        engine=None,
        **kwargs
    ) -> None:
        super().__init__(name="Syntax Error Feedback", **kwargs)
        self.engine = engine,
        self.repl = PythonREPL()
        self.type = "tool"

    def runPythonWithTM(self, soln):
        try:
            return func_timeout.func_timeout(60, self.repl.run, args=(soln,))
        except func_timeout.FunctionTimedOut:
            print('Running python solution timed out')
        return ""

    def __call__(self, solutions: List[str], **kwargs) -> List[str]:
        """
        Returns first syntax error in the solution or empty string if no syntax error
        Args:
            solutions (List[str]): list of solutions
        Returns:
            List[str]: list of feedbacks
        """
        executable_solutions = [solution +
                                "\n(solution())" for solution in solutions]
        fb_and_maybe_slns = [{"feedback": self.runPythonWithTM(
            solution), "solution": solution} for solution in executable_solutions]
        return fb_and_maybe_slns


@FeedbackFactory.register("self_refine")
class SelfRefineFeedback(LLMFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Original Self-Refine", max_tokens=600,
                         answer_prefix="def solution():", eager_refine=True, **kwargs)
        self.type = "lm"
        self.instruction = "# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("missing_step")
class MissingStepFeedback(LLMFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Missing Step Feedback", max_tokens=300,
                         answer_prefix="def solution():", **kwargs)
        self.type = "lm"
        self.instruction = """# Check each semantically complete block of code for any missing steps and suggest the correct way to add them. Ignore all the other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("variable_naming")
class VariableNameFeedback(LLMFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Variable Naming Feedback", max_tokens=600,
                          answer_prefix="def solution():", **kwargs)
        self.type = "lm"
        self.instruction = """# Check each semantically complete block of code and identify the variables that are not named correctly or may cause confusion and fix the issues. State the assumptions you made when renaming the variables clearly. Ignore all the other type of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("commonsense")
class CommonsenseFeedback(LLMFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs,
    ) -> None:
        super().__init__(name="Commonsense Feedback", max_tokens=300,
                         answer_prefix="def solution():", **kwargs)
        self.type = "lm"
        self.instruction = """# Check each semantically complete block of the code to check for any commonsense errors. Commonsense reasoning errors are errors about any relation or knowledge that is should be known from general world such as "all ducks are birds". State the assumptions you made clearly. Ignore all the other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("hallucination")
class HallucinationFeedback(LLMFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Hallucination Feedback", max_tokens=300,
                         answer_prefix="def solution():", **kwargs)
        self.type = "lm"
        self.instruction = """# Check each semantically complete block of code for any hallucination errors and suggest fixes. Hallucination errors are steps that are supported by neither the context nor the real world. Ignore all other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("coherency")
class CoherencyFeedback(LLMFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Coherency Feedback", max_tokens=300,
                         answer_prefix="def solution():", **kwargs)
        self.type = "lm"
        self.instruction = """# Check the code for any coherency errors and suggest fixes. Coherency errors are steps that contradict each other or do not follow a cohesive story. Ignore all other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("redundancy")
class RedundancyFeedback(LLMFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Redundancy Feedback", max_tokens=300,
                         answer_prefix="def solution():", **kwargs)
        self.type = "lm"
        self.instruction = """# Check each semantically complete block of code for any redundancy errors and suggest fixes. Redundancy errors are steps that contain redundant information, which even though might be factual, is not required to answer the question. Ignore all other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("missing_step_os")
class MissingStepFeedbackOS(OSFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Missing Step Feedback", max_tokens=300,
                         answer_prefix="def solution():", **kwargs)
        self.type = "lm"
        self.instruction = """# Check each semantically complete block of code for any missing steps and suggest the correct way to add them. Ignore all the other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("variable_naming_os")
class VariableNameFeedbackOS(OSFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Variable Naming Feedback", max_tokens=900,
                         eager_refine=True, answer_prefix="def solution():", **kwargs)
        self.type = "lm"
        self.instruction = """# Check each semantically complete block of code and identify the variables that are not named correctly or may cause confusion and fix the issues. State the assumptions you made when renaming the variables clearly. Ignore all the other type of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("commonsense_os")
class CommonsenseFeedbackOS(OSFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs,
    ) -> None:
        super().__init__(name="Commonsense Feedback", max_tokens=300,
                         answer_prefix="def solution():", **kwargs)
        self.type = "lm"
        self.instruction = """# Check each semantically complete block of the code to check for any commonsense errors. Commonsense reasoning errors are errors about any relation or knowledge that is should be known from general world such as "all ducks are birds". State the assumptions you made clearly. Ignore all the other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("hallucination_os")
class HallucinationFeedbackOS(OSFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Hallucination Feedback", max_tokens=600,
                         answer_prefix="def solution():", **kwargs)
        self.type = "lm"
        self.instruction = """# Check each semantically complete block of code for any hallucination errors and suggest fixes. Hallucination errors are steps that are supported by neither the context nor the real world. Ignore all other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("coherency_os")
class CoherencyFeedbackOS(OSFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Coherency Feedback", max_tokens=300,
                         answer_prefix="def solution():", **kwargs)
        self.type = "lm"
        self.instruction = """# Check the code for any coherency errors and suggest fixes. Coherency errors are steps that contradict each other or do not follow a cohesive story. Ignore all other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)


def test():
    missing_step = FeedbackFactory.create_feedback(
        'missing_step', engine='text-davinci-003', temperature=0.7, prompt_examples='prompt/gsm_maf/missing_step.txt')
    variable_naming = FeedbackFactory.create_feedback(
        'variable_naming', engine='text-davinci-003', temperature=0.7, prompt_examples='prompt/gsm?maf/variable_naming_eager.txt')
    logical = FeedbackFactory.create_feedback(
        'logical', engine='text-davinci-003', temperature=0.7, prompt_examples='prompt/gsm_maf/logical.txt')

    wrong_solns = ["""def solution():
    \"\"\"Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?\"\"\"
    chips_per_inch = 12
    chips_per_bag = 72
    bags = 2
    height = 3
    chips_needed = height * chips_per_inch
    chips_left = chips_available - chips_needed
    length = chips_left / chips_per_inch
    result = length
    return result""",
                   "def solution():\n    \"\"\"Helga went shopping for a new pair of shoes. At the first store, she tried on 7 pairs of shoes. At the second store, she tried on 2 more pairs than at the first store. At the third store, she did not try on any shoes, but she did buy a scarf. But at the fourth store, she tried on twice as many pairs of shoes as she did at all three other stores combined, before finally choosing a pair to buy. Helga's neighbor tried on 20 pairs of pants than Helga. What is the total number of pairs of shoes Helga tried on before buying her new shoes?\"\"\"\n    shoes_first_store = 7\n    shoes_second_store = shoes_first_store + 2\n    shoes_third_store = 0\n    shoes_fourth_store = 2 * (shoes_first_store + shoes_second_store + shoes_third_store)\n    total_shoes_tried_on = shoes_first_store + shoes_second_store + shoes_third_store + shoes_fourth_store\n    neighbor_pants = total_shoes_tried_on + 20\n    result = total_shoes_tried_on\n    return result"
                   ]

    # ----- OpenAI Engines ----- #
    # vn_feedback_and_solns = variable_naming(wrong_solns)
    # for i, vn_feedback_and_soln in enumerate(vn_feedback_and_solns):
    #     print(f"Variable Naming Feedback {i}:\n{vn_feedback_and_soln['feedback']}")
    #     print(f"Variable Naming Solution {i}:\n{vn_feedback_and_soln['solution']}")

    # ms_feedbacks = missing_step([x['solution'] for x in vn_feedback_and_solns])
    # print(len(ms_feedbacks))
    # for i, ms_feedback in enumerate(ms_feedbacks):
    #     print(f"Missing Step Feedback {i}:\n{ms_feedback}")

    # logical_feedbacks = logical([x['solution'] for x in vn_feedback_and_solns])
    # for i, logical_feedback in enumerate(logical_feedbacks):
    #     print(f"Logical Feedback {i}:\n{logical_feedback}")

    # ----- OS Engines ----- #
    variable_naming_os = FeedbackFactory.create_feedback(
        'variable_naming_os', engine='vicuna', temperature=0.0, prompt_examples='prompt/gsm?maf/variable_naming_eager_os.txt')
    vn_feedback_and_solns = variable_naming_os(wrong_solns)
    for i, vn_feedback_and_soln in enumerate(vn_feedback_and_solns):
        print(
            f"Variable Naming Feedback {i}:\n{vn_feedback_and_soln['feedback']}")
        print(
            f"OS Variable Naming Solution {i}:\n{vn_feedback_and_soln['solution']}")

    del variable_naming_os
    torch.cuda.empty_cache()

    missing_step_os = FeedbackFactory.create_feedback(
        'missing_step_os', engine='vicuna', temperature=0.0, prompt_examples='prompt/gsm_maf/missing_step_os.txt')
    ms_feedbacks = missing_step_os([x['solution']
                                   for x in vn_feedback_and_solns])
    print(len(ms_feedbacks))
    for i, ms_feedback in enumerate(ms_feedbacks):
        print(f"OS Missing Step Feedback {i}:\n{ms_feedback}")

    del ms_feedbacks
    del missing_step_os
    torch.cuda.empty_cache()

    logical_os = FeedbackFactory.create_feedback(
        'logical_os', engine='vicuna', temperature=0.0, prompt_examples='prompt/gsm_maf/logical_os.txt')
    logical_feedbacks = logical_os([x['solution']
                                   for x in vn_feedback_and_solns])
    for i, logical_feedback in enumerate(logical_feedbacks):
        print(f"OS Logical Feedback {i}:\n{logical_feedback}")

    del logical_feedbacks
    del logical_os
    torch.cuda.empty_cache()


def test_syntax():
    print('entering test_syntax')
    wrong_solns = ["""def solution():
    \"\"\"Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?\"\"\"
    chips_per_inch = 12
    chips_per_bag = 72
    bags = 2
    height = 3
    chips_needed = height * chipsj_per_inch
    chips_left = chips_available - chips_needed
    length = chips_left / chips_per_inch
    result = length
    return result"""]
    syntax = PythonExecuter()
    syntax_feedbacks = syntax(wrong_solns)
    print(syntax_feedbacks)


if __name__ == '__main__':
    test_syntax()
