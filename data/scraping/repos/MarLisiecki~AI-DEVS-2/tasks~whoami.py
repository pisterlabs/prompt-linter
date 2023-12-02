from typing import Tuple

from open_ai_connector.const import OpenAiModels
from open_ai_connector.open_ai_connector import OpenAIConnector
from solver.prompt_builder import prepare_prompt
from solver.solver import Solver

ASSISTANT_CONTENT = (
    "Guess the person based on the information that can be found in context {context}. If you need more "
    "information just say NO, if you know the answer say YES write name and surname of the person without any additional"
    "information - ultra-briefly"
)
USER_CONTENT = "{question}"


def prepare_context(context: str, hint: str = "") -> Tuple[str, str]:
    if hint != "":
        context += " " + hint
    oai = OpenAIConnector()
    prompt = prepare_prompt(
        ASSISTANT_CONTENT.format(context=context),
        USER_CONTENT.format(question="Do you know the answer?"),
    )
    answer = oai.generate_answer(model=OpenAiModels.gpt4.value, messages=prompt)
    return answer, context


def whoami(input_data: dict) -> dict:
    answer, context = prepare_context(input_data["hint"])
    while "NO" in answer:
        print("Not enough information!")
        sol.download_input_data()
        another_hint = sol.input_data["hint"]
        answer, context = prepare_context(context, another_hint)
    else:
        prepared_answer = {"answer": answer}
    return prepared_answer


if __name__ == "__main__":
    sol = Solver("whoami")
    sol.solve(whoami)
