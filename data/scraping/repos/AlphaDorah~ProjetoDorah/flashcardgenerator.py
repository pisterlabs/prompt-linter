"""
Utilizar:

`python -m integration.test_flashcardgenerator_maritalk`

Para executar o teste de integração:

`integration/test_flashcardgenerator_maritalk.py`
"""

from abc import ABC, abstractmethod
from langchain.prompts import PromptTemplate
from src.dorahLLM.flashcard.flashcard import Flashcard

from src.dorahLLM.maritalkllm import MariTalkLLM
from langchain.chains import LLMChain
from langchain.llms.base import LLM


class FlashcardGenerator(ABC):
    @abstractmethod
    def __init__(self, model, template: str):
        pass

    @abstractmethod
    def generate(self) -> str:
        pass


class MaritalkFlashcardGenerator(FlashcardGenerator):
    def __init__(self, model: LLM = MariTalkLLM(), template="maritalk_flashcard"):
        self.template = PromptTemplate.from_template(self._load_template(template))
        self.model = model
        self.chain = LLMChain(prompt=self.template, llm=self.model)

    def generate(self, summary: str) -> list[Flashcard]:
        res = self.chain(inputs={"summary": summary})
        return self._parse_flashcards(res["text"])

    def _parse_flashcards(self, text: str) -> list[Flashcard]:
        print(text)
        flashcards = []
        lines = text.split("\n")
        questions = []
        answers = []
        for line in lines:
            if line.startswith("Pergunta: "):
                questions.append(line.removeprefix("Pergunta: "))
            elif line.startswith("Resposta: "):
                answers.append(line.removeprefix("Resposta: "))

        if len(questions) == len(answers):
            for i in range(len(questions)):
                flashcards.append(Flashcard(questions[i], answers[i]))
        return flashcards

    def _load_template(self, template: str) -> str:
        with open("./src/dorahLLM/flashcard/res/" + template + ".txt") as f:
            template = f.read()
        return template
