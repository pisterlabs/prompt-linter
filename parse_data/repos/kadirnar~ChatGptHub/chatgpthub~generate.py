from typing import Optional

from chatgpthub import (
    load_openai_key,
    load_promptlayer_key,
    prompt_template,
    promptlayer_chatgpt,
    translate_chatgpt,
)


def load_key(
    openai_key: Optional[str] = None, promptlayer_key: Optional[str] = None
):
    """
    This function loads the OpenAI and PromptLayer keys.
    Args:
        openai_key: The OpenAI key to use. Defaults to None.
        promptlayer_key: The PromptLayer key to use. Defaults to None.
    """
    if openai_key is not None:
        load_openai_key(openai_key)
    if promptlayer_key is not None:
        load_promptlayer_key(promptlayer_key)


class ChatGptHubDemo:
    def __init__(
        self,
        openai_key: Optional[str] = None,
        promptlayer_key: Optional[str] = None,
    ):
        """
        This class is a demo for the chatgpthub library.
        Args:
            openai_key: The OpenAI key to use. Defaults to None.
            promptlayer_key: The PromptLayer key to use. Defaults to None.
        """
        load_key(openai_key=openai_key)

        self.promptlayer_key = promptlayer_key

    def translate(
        self,
        model_name: str = "gpt-3.5-turbo",
        input_language: str = "English",
        output_language: str = "Turkish",
        text: str = "Hello, how are you?",
        temperature: float = 0.0,
    ):
        """
        This function is a demo for the translate_chatgpt function.
        Args:
            model_name: The name of the model to use. Defaults to "gpt-3.5-turbo".
            input_language: The language to translate from. Defaults to "English".
            output_language: The language to translate to. Defaults to "Turkish".
            text: The text to translate. Defaults to "Hello, how are you?".
            temperature: The temperature to use for the model. Defaults to 0.0.
        Returns:
            The translated text.
        """
        output = translate_chatgpt(
            model_name=model_name,
            input_language=input_language,
            output_language=output_language,
            text=text,
            temperature=temperature,
        )

        return output

    def custom_template(
        self,
        model_name: str = "gpt-3.5-turbo",
        template: str = "You are a helpful assistant that English to Turkish and you are asked to translate the following text: {text}",
        input_variables: str = "text",
        text: str = "Hello, how are you?",
        temperature: float = 0.0,
    ):
        """
        This function is a demo for the prompt_template function.
        Args:
            model_name: The name of the model to use. Defaults to "gpt-3.5-turbo".
            template: The prompt template to use. Defaults to "You are a helpful assistant that English to Turkish and you are asked to translate the following text: {text}".
            input_variables: The input variables to use. Defaults to "text".
            text: The text to translate. Defaults to "Hello, how are you?".
            temperature: The temperature to use for the model. Defaults to 0.0.
        Returns:
            The translated text.
        """
        output = prompt_template(
            model_name=model_name,
            template=template,
            input_variables=input_variables,
            text=text,
            temperature=temperature,
        )

        return output

    def promptlayer(
        self,
        model_name: str = "gpt-3.5-turbo",
        text: str = "Hello, I am a chatbot and",
        temperature: float = 0.0,
    ):
        """
        This function is a demo for the promptlayer_chatgpt function.
        Args:
            model_name: The name of the model to use. Defaults to "gpt-3.5-turbo".
            text: The text to use as the prompt. Defaults to "Hello, I am a chatbot and".
            temperature: The temperature to use for the model. Defaults to 0.0.
        Returns:
            The chatbot's response.
        """
        if self.promptlayer_key is not None:
            load_promptlayer_key(self.promptlayer_key)
        else:
            raise Exception(
                "You need to provide a PromptLayer key to use the promptlayer_chatgpt function."
            )

        output = promptlayer_chatgpt(
            model_name=model_name, text=text, temperature=temperature
        )

        return output
