import os
from functools import lru_cache

import streamlit as st
import anthropic
import openai


# st.set_page_config(layout="wide")


class AnthropicSerivce:
    def __init__(self):
        if "ANTHROPIC_API_KEY" not in os.environ:
            os.environ["ANTHROPIC_API_KEY"] = st.text_input("Anthropic API Key", type="password")
        self.client = self.anthropic_client()

    @staticmethod
    @lru_cache
    def anthropic_client():
        return anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

    def prompt(self, prompt, max_tokens_to_sample: int = 2000):
        response = self.client.completion_stream(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            max_tokens_to_sample=max_tokens_to_sample,
            model="claude-v1",
            stream=True,
        )
        return response


class OpenAIService:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def list_models(self):
        return openai.Model.list()

    def prompt(self, prompt):
        # return openai.Completion.create(model="ada", prompt=prompt)
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
            # model="gpt-4", messages=[{"role": "user", "content": prompt}]
        )


def app():
    prompt = st.text_area("Prompt", value="How many toes do dogs have?", height=100)
    tabs = st.tabs(["Anthropic", "OpenAI"])
    if prompt:
        # -- Antrhopic
        with tabs[0]:
            anthropic_service = AnthropicSerivce()
            response = anthropic_service.prompt(prompt=prompt)
            # for data in response:
            #     col = st.columns(2)
            #     with col[0]:
            #         st.markdown("---")
            #         st.write(data)
            #     with col[1]:
            #         st.markdown("---")
            #         st.markdown(data["completion"])
            # -- pick the last response

            col = st.columns(2)
            answer = list(response)[-1]
            with col[0]:
                st.write(answer)
            with col[1]:
                st.code(answer["completion"])

        # -- OpenAI
        with tabs[1]:
            openai_service = OpenAIService()
            response = openai_service.prompt(prompt=prompt)

            col = st.columns(2)
            # st.write(openai_service.list_models())
            with col[0]:
                st.write(response)
                st.markdown("---")
            with col[1]:
                answer = response.choices[0]["message"]["content"]
                st.code(answer, language="python")
                st.markdown("---")


if __name__ == "__main__":

    app()
