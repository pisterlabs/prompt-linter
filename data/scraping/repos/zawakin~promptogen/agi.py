from typing import Callable, Tuple

import promptogen as pg
from examples.llm.openai_util import OpenAITextLLM
from promptogen.model.dataclass import DataClass
from promptogen.prompt_collection import PromptCollection, PromptCreatorPrompt
from promptogen.prompt_collection.prompts.text_summarizer import TextSummarizerPrompt
from promptogen.prompt_interceptor.translation_interceptor import ValueTranslationInterceptor

formatter = pg.KeyValuePromptFormatter()
llm = OpenAITextLLM(model="gpt-3.5-turbo")

prompt_runner = pg.TextLLMPromptRunner(llm=llm, formatter=formatter)

prompt_creator_prompt = PromptCreatorPrompt()

signatures = [
    (
        "model_restructure_problem",
        "Based on the initial description, restructure and provide a more detailed version of the problem, including relevant context if possible.",
        "(initial_description: str) -> (detailed_description: str, context: str)",
    ),
    (
        "model_refine_with_feedback",
        "Consider the user feedback and refine the problem definition. Output suggested modifications.",
        "(detailed_description: str, user_feedback: str) -> (refined_description: str)",
    ),
    (
        "model_final_approach",
        "Based on the iterative feedback and adjustments, finalize the problem definition.",
        "(refined_description: str, feedback_history: List[str]) -> (final_problem_definition: str)",
    ),
]


def setup_prompts():
    prompts = {}

    for name, description, signature in signatures:
        resp = prompt_runner.run_prompt(
            prompt_creator_prompt,
            input_value={
                "description": description,
                "background": signature,
            },
        )

        prompt = pg.Prompt.from_dict(resp["prompt"])
        prompts[name] = prompt

    PromptCollection(prompts=prompts).to_json_file("agi_like.json")


setup_prompts()
# exit()


prompts = PromptCollection.from_json_file("agi_like.json")

ps = prompts.prompts


from typing import List


def model_restructure_problem(initial_description: str) -> Tuple[str, str]:
    resp = prompt_runner.run_prompt(
        ps["model_restructure_problem"],
        input_value={
            "initial_description": initial_description,
        },
    )
    return resp["detailed_description"], resp["context"]


def model_refine_with_feedback(detailed_description: str, user_feedback: str) -> str:
    resp = prompt_runner.run_prompt(
        ps["model_refine_with_feedback"],
        input_value={
            "detailed_description": detailed_description,
            "user_feedback": user_feedback,
        },
    )
    return resp["refined_description"]


def model_final_approach(refined_description: str, feedback_history: List[str]) -> str:
    resp = prompt_runner.run_prompt(
        ps["model_final_approach"],
        input_value={
            "refined_description": refined_description,
            "feedback_history": feedback_history,
        },
    )
    return resp["final_problem_definition"]


def define_problem_with_model():
    print("Provide an initial description of a problem you perceive.")
    initial_description = input("> ")

    detailed_description, context = model_restructure_problem(initial_description)
    print(f"Model's Refined Problem: {detailed_description}")
    print(f"Model's Context: {context}")

    feedback_history = []

    while True:
        print("Provide feedback on the refined problem (or type 'done' to finalize):")
        user_feedback = input("> ")

        if user_feedback.lower() == "done":
            break

        feedback_history.append(user_feedback)
        detailed_description = model_refine_with_feedback(detailed_description, user_feedback)
        print(f"Model's Refined Problem with Feedback: {detailed_description}")

    final_problem_definition = detailed_description
    print(f"Final Problem Definition: {final_problem_definition}")

    result = model_final_approach(final_problem_definition, feedback_history)
    print(f"Suggested approach: {result}")


define_problem_with_model()
