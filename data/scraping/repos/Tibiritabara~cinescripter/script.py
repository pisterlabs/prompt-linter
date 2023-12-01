import openai

from services.embedding import Embeddings
from services.weaviate import Weaviate
from utils.common import SettingsLoader
from utils.logs import logger


class Script:

    APP_NAME = "OPENAI_SCRIPT"

    def __init__(self, **kwargs):
        self.options = SettingsLoader.load(
            self.APP_NAME,
            kwargs
        )

    def _create_paragraphs(self, answers):
        return '\n'.join(answers)

    def get_context(self, question: str) -> str:
        question_embedding = Embeddings().embed(question)
        context = self._create_paragraphs(Weaviate().get_answers(question_embedding))
        return context

    def get_script(self, context: str, question: str, tone: str) -> str:
        prompt = self.options.get("prompt").substitute({
            "tone": tone,
            "question": question,
        })
        logger.debug("OpenAI parameters %s", {"tone": tone, "question": question, "prompt": prompt, "context": context})

        response = openai.ChatCompletion.create(
            model=self.options.get("model"),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": context},
            ]
        )

        return response["choices"][0]["message"]["content"]
