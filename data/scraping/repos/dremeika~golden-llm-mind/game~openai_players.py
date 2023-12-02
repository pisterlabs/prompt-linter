from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
)

from model import Player, R1Answer, R1Question, R2Question, R2Answer, R3Question, R3Answer, R4Question, R4Answer

q1_prompt_template = ChatPromptTemplate.from_messages([
    "Tu esi protmūšio dalyvis. Tavo užduotis yra atsakyti į klausimus",
    """
    Klausimas: {question}
    Atsakymo variantai: {options}
    Atsakymas: """
])

q2_prompt_template = ChatPromptTemplate.from_messages([
    "Tu esi protmūšio dalyvis. Tavo užduotis yra pateikti atsakymą pasinaudojant užuominomis",
    """
    Tema: {question}
    Užuominos:
    {hints}
    Atsakymas: """
])

q3_prompt_template = ChatPromptTemplate.from_messages([
    "Tu esi protmūšio dalyvis. Tavo užduotis yra pasirinkti vieną iš dviejų atsakymų"
    """
    Klausimas: {question} {query}
    Galimi atsakymai: {choices}
    Atsakymas: """
])

q4_prompt_template = ChatPromptTemplate.from_messages([
    "Tu esi protmūšio dalyvis. Tavo užduotis yra teisingai atsakyti į klausimą",
    """
    Klausimas: {question}
    Atsakymas: """
])


class GPTPlayer(Player):

    def __init__(self, model_name: str) -> None:
        self.chat = ChatOpenAI(temperature=0, model_name=model_name)

    def play_round1(self, question: R1Question) -> R1Answer:
        prediction = self.chat(
            q1_prompt_template
            .format_prompt(question=question.question, options=", ".join(question.options))
            .to_messages())
        return R1Answer(question=question, answer=prediction.content)

    def play_round2(self, question: R2Question, hints: list[str]) -> R2Answer:
        prediction = self.chat(
            q2_prompt_template
            .format_prompt(question=question.question, hints="\n".join(hints))
            .to_messages())
        return R2Answer(question=question, answer=prediction.content)

    def play_round3(self, question: R3Question, query: str) -> R3Answer:
        prediction = self.chat(
            q3_prompt_template
            .format_prompt(question=question.question, query=query, choices=", ".join(question.choices))
            .to_messages())
        return R3Answer(question=question, answer=prediction.content)

    def play_round4(self, question: R4Question) -> R4Answer:
        prediction = self.chat(
            q4_prompt_template
            .format_prompt(question=question.question)
            .to_messages())
        return R4Answer(question=question, answer=prediction.content)