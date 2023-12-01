from open_ai_connector.open_ai_connector import OpenAIConnector
from solver.solver import Solver

ASSISTANT_CONTENT = ""
USER_CONTENT = ""


def gnome(input_data: dict) -> dict:
    url = input_data["url"]
    assistant_knowledge = 'Answer ultra briefly, return only color in Polish, if there is no hat on image return string "error". '
    oai = OpenAIConnector()
    answer = oai.use_vision(
        text="What is the color of gnome hat",
        url=url,
        assistant_knowledge=assistant_knowledge,
    )
    prepared_answer = {"answer": answer}
    return prepared_answer


if __name__ == "__main__":
    sol = Solver("gnome")
    sol.solve(gnome)
