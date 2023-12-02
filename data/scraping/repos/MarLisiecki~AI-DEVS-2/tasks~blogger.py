from open_ai_connector.const import OpenAiModels
from open_ai_connector.open_ai_connector import OpenAIConnector
from solver.prompt_builder import prepare_prompt
from solver.solver import Solver


ASSISTANT_CONTENT = (
    "You are a blogger who will prepare a short post about certain topic"
)
USER_CONTENT = "Write concise post about the topic: {sentence} Return answer in Polish"


def blogger(input_data: dict) -> dict:
    chapters = input_data["blog"]
    oai = OpenAIConnector()
    answers = []
    for chapter in chapters:
        prompt = prepare_prompt(
            ASSISTANT_CONTENT, USER_CONTENT.format(sentence=chapter)
        )
        answer = oai.generate_answer(
            model=OpenAiModels.gpt3_5_turbo.value, messages=prompt
        )
        answers.append(answer)
    prepared_answer = {"answer": answers}
    return prepared_answer


if __name__ == "__main__":
    sol = Solver("blogger")
    sol.solve(blogger)
