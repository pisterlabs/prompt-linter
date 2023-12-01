import streamlit as st
import time
from helper import get_var
from openai import OpenAI
import tiktoken

MAX_ERRORS = 3
LLM_RETRIES = 3
SLEEP_TIME_AFTER_ERROR = 30
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 500
MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]
MODEL_TOKEN_PRICING = {
    "gpt-3.5-turbo": {"in": 0.0015, "out": 0.002},
    "gpt-4": {"in": 0.03, "out": 0.06},
}


class ToolBase:
    def __init__(self, logger):
        self.logger = logger
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.model = MODEL_OPTIONS[0]
        self.tokens_in = 0
        self.tokens_out = 0

    def get_model(self):
        return st.selectbox(
            "Model",
            options=MODEL_OPTIONS,
            index=0,
            help="Wählen Sie das LLM Modell, das Sie verwenden möchten.",
        )

    def get_intro(self):
        """
        Reads the markdown content from a file with the same name as the script and returns it.

        Returns:
        str: The markdown content of the file.
        """
        with open(f"{self.script_name}.md", "r", encoding="utf-8") as file:
            markdown_content = file.read()
        return markdown_content

    def token_use_expression(self, tokens: list):
        cost_tokens_in = MODEL_TOKEN_PRICING[self.model]["in"] * tokens[0] / 1000
        cost_tokens_out = MODEL_TOKEN_PRICING[self.model]["out"] * tokens[1] / 1000
        return f"""
            Tokens in: {tokens[0]} Kosten: ${cost_tokens_in: .2f}\n
            Tokens out: {tokens[1]} Kosten: ${cost_tokens_out: .2f}\n
            Total Tokens: {tokens[0] + tokens[1]} Kosten: ${(cost_tokens_in + cost_tokens_out): .2f}
            """

    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def get_completion(self, text, index):
        """Generates a response using the OpenAI ChatCompletion API based on
        the given text.

        Args:
            text (str): The user's input.

        Returns:
            str: The generated response.

        Raises:
            None
        """
        client = OpenAI(
            api_key=get_var("OPENAI_API_KEY"),
        )
        retries = LLM_RETRIES
        while retries > 0:
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                tokens = [
                    completion.usage.completion_tokens,
                    completion.usage.prompt_tokens,
                ]
                return completion.choices[0].message.content, tokens
            except Exception as err:
                st.error(f"OpenAIError {err}, Index = {index}")
                retries -= 1
                time.sleep(SLEEP_TIME_AFTER_ERROR)
        return [], 0

    def show_settings(self):
        pass

    def show_ui(self):
        """
        Displays the user interface for the tool, which includes three tabs:
        'Input and Settings', 'Run', and 'Information'. The 'Input and Settings'
        tab displays the tool's settings, the 'Run' tab executes the tool, and the
        'Information' tab displays introductory text about the tool.
        """
        st.subheader(self.title)
        tabs = st.tabs(["⚙️Input und Einstellungen", "🚀Ausführen", "💁Informationen"])
        with tabs[0]:
            self.show_settings()
        with tabs[1]:
            self.run()
        with tabs[2]:
            text = self.intro
            st.markdown(text, unsafe_allow_html=True)

    def run(self):
        pass
