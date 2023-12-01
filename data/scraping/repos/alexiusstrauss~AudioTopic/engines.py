from http import HTTPStatus

import requests
from langchain.llms.openai import OpenAI
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

from src.services.exceptions import ApiKeyException
from src.summarization.interfaces import Summarization


class LangChain(Summarization):
    def __init__(self, api_key: str):
        self.llm = OpenAI(temperature=0, api_key=api_key)
        self.api_key = api_key

    def summarize(self, text: str) -> str:
        prompt = f"Por favor, analise o seguinte texto e crie um resumo muito breve, \
                  apenas com as informações mais relevantes. \
                  O resumo deve ser conciso, com poucas palavras, \
                  destacando apenas os aspectos mais importantes do texto. \
                  Aqui está o texto para análise: \n\n{text}\n\n Sumário:"
        return self.llm.predict(prompt)

    def token_is_valid(self):
        url = "https://api.openai.com/v1/engines/davinci-codex/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "prompt": "Translate the following English text to French: 'Hello, world!'",
            "max_tokens": 10,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=10,
            )
        except Exception as exc:
            raise ApiKeyException() from exc

        if response.status_code in [
            HTTPStatus.UNAUTHORIZED,
            HTTPStatus.BAD_REQUEST,
            HTTPStatus.FORBIDDEN,
        ]:
            raise ApiKeyException()


class TensorFlow(Summarization):
    def __init__(self, model_name="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text: str, max_length=650):
        # Preparar a entrada para o modelo
        inputs = self.tokenizer.encode(f"summarize: {text}", return_tensors="tf", max_length=1600, truncation=True)
        # Gerar a saída do modelo
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            length_penalty=4.0,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
