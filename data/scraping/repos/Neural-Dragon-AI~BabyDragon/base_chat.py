from typing import Callable, Dict, Generator, List, Tuple, Union

import tiktoken
from IPython.display import Markdown, display
from pydantic import BaseModel, validator

from babydragon.chat.prompts.default_prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
)
from babydragon.memory.threads.base_thread import BaseThread
from babydragon.models.generators.chatgpt import chatgpt_response
from babydragon.models.generators.cohere import cohere_response
from babydragon.utils.chatml import (
    get_mark_from_response,
    get_str_from_response,
    mark_question,
    mark_system,
)


class PromptConfiguration(BaseModel):
    system_prompt: Union[str, None] = None
    user_prompt: Union[str, None] = None

    @validator("system_prompt", pre=True, always=True)
    def validate_system_prompt(cls, value):
        return value if value is not None else DEFAULT_SYSTEM_PROMPT

    @validator("user_prompt", pre=True, always=True)
    def validate_user_prompt(cls, value):
        return value if value is not None else DEFAULT_USER_PROMPT


class MessageHandler(BaseModel):
    content: str

    @validator("content", pre=True, always=True)
    def validate_content(cls, value):
        return value if value is not None else ""


class Prompter:
    def __init__(
        self,
        system_prompt: Union[str, None] = None,
        user_prompt: Union[str, None] = None,
    ):
        config = PromptConfiguration(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        self.system_prompt = config.system_prompt
        self.user_prompt_template = config.user_prompt
        self.prompt_func: Callable[[str], Tuple[List[str], str]] = self.one_shot_prompt

    def set_default_prompts(self) -> None:
        """
        Set the default system and user prompts.
        """
        config = PromptConfiguration(system_prompt=None, user_prompt=None)
        self.system_prompt = config.system_prompt
        self.user_prompt_template = config.user_prompt

    def default_user_prompt(self, message: str) -> str:
        """
        Compose the default user prompt.

        :param message: A string representing the user message.
        """
        return self.user_prompt_template.format(question=message)

    def one_shot_prompt(self, message: str) -> Tuple[List[str], str]:
        """
        Compose the prompt for the chat-gpt API.

        :param message: A string representing the user message.
        :return: A tuple containing a list of strings representing the prompt and a string representing the marked question.
        """
        message = MessageHandler(content=message).content
        marked_question = mark_question(self.default_user_prompt(message))
        prompt = [mark_system(self.system_prompt)] + [marked_question]
        return prompt, marked_question

    def one_shot_prompt_with_thread(
        self, message: str, thread: BaseThread
    ) -> Tuple[List[str], str]:
        """
        Compose the prompt for the chat-gpt API, including conversation history from the thread.

        :param message: A string representing the user message.
        :param thread: An instance of BaseThread containing conversation history.
        :return: A tuple containing a list of strings representing the prompt and a string representing the marked question.
        """
        # Validate and handle the user's current message
        message = MessageHandler(content=message).content
        marked_question = mark_question(self.default_user_prompt(message))

        # Retrieve conversation history from the thread (modify as needed to get the desired context)
        conversation_history = thread.memory_thread

        # Construct the prompt using the system prompt, conversation history, and marked question
        prompt = [mark_system(self.system_prompt)]
        for index in range(len(conversation_history)):
            role = conversation_history["role"][index]
            content = conversation_history["content"][index]
            prompt.append(
                mark_system(content) if role == "system" else mark_question(content)
            )

        prompt.append(marked_question)

        return prompt, marked_question

    def update_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt.

        :param new_prompt: A string representing the new system prompt.
        """
        self.system_prompt = PromptConfiguration(system_prompt=new_prompt).system_prompt

    def update_user_prompt(self, new_prompt: str) -> None:
        """
        Update the user prompt.

        :param new_prompt: A string representing the new user prompt.
        """
        self.user_prompt_template = PromptConfiguration(
            user_prompt=new_prompt
        ).user_prompt


class ChatModelConfiguration(BaseModel):
    model: str = "gpt-3.5-turbo"
    max_output_tokens: int = 200


class BaseChat:
    def __init__(self, model: Union[str, None] = None, max_output_tokens: int = 200):
        """
        Initialize the BaseChat with a model and max_output_tokens.

        :param model: A string representing the chat model to be used.
        :param max_output_tokens: An integer representing the maximum number of output tokens.
        """
        config = ChatModelConfiguration(
            model=model, max_output_tokens=max_output_tokens
        )
        self.model = config.model
        self.max_output_tokens = config.max_output_tokens

        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.reset_logs()
        self.prompt_func = self.identity_prompter

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def identity_prompter(self, message: str) -> Tuple[List[Dict], str]:
        """
        A simple identity prompter that takes a message and returns the message marked as a question.

        :param message: A string representing the user message.
        :return: A tuple containing the marked question and the original message.
        """
        return [mark_question(message)], mark_question(message)

    def chat_response(
        self,
        prompt: List[dict],
        max_tokens: Union[int, None] = None,
        stream: bool = False,
    ) -> Union[Generator, Tuple[Dict, bool]]:
        if max_tokens is None:
            max_tokens = self.max_output_tokens

        if "gpt" in self.model:
            response, status = chatgpt_response(
                prompt=prompt, model=self.model, max_tokens=max_tokens, stream=stream
            )
            return response, status if status else self.handle_failure(response)

        elif "command" in self.model:
            response, status = cohere_response(
                prompt=prompt, model=self.model, max_tokens=max_tokens
            )
            return response, status if status else self.handle_failure(response)

        else:
            return {}, False

    def reply(
        self, message: str, verbose: bool = True, stream: bool = False
    ) -> Union[Generator, str]:
        """
        Reply to a given message using the chatbot.

        :param message: A string representing the user message.
        :return: A string representing the chatbot's response.
        """
        if stream:
            return self.query(message, verbose, stream)
        else:
            return self.query(message, verbose)["content"]

    def query(
        self, message: str, verbose: bool = True, stream: bool = False
    ) -> Union[Generator, str]:
        """
        Query the chatbot with a given message, optionally showing the input and output messages as Markdown.

        :param message: A string representing the user message.
        :param verbose: A boolean indicating whether to display input and output messages as Markdown.
        :return: A string representing the chatbot's response.
        """

        prompt, _ = self.prompt_func(message)

        if stream:
            return self.chat_response(prompt=prompt, stream=stream)

        response, success = self.chat_response(prompt)
        if verbose:
            display(Markdown("#### Question: \n {question}".format(question=message)))
        if success:
            answer = get_mark_from_response(response, self.model)
            self.outputs.append(answer)
            self.inputs.append(message)
            self.prompts.append(prompt)
            if verbose:
                display(
                    Markdown(
                        " #### Anwser: \n {answer}".format(
                            answer=get_str_from_response(response, self.model)
                        )
                    )
                )
            return answer
        else:
            raise Exception("OpenAI API Error inside query function")

    def handle_failure(self, response):
        self.failed_responses.append(response)
        return response, False

    def reset_logs(self):
        """
        Reset the chatbot's memory.
        """
        self.outputs = []
        self.inputs = []
        self.prompts = []
        self.failed_responses = []

    def get_logs(self):
        """
        Get the chatbot's memory.

        :return: A tuple containing the chatbot's memory as three lists of strings.
        """
        return self.inputs, self.outputs, self.prompts

    def run_text(
        self, text: str, state: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Process the user's text input and update the chat state.

        :param text: A string representing the user input.
        :param state: A list of tuples representing the current chat state.
        :return: A tuple containing the updated chat state as two lists of tuples.
        """
        print("===============Running run_text =============")
        print("Inputs:", text)
        try:
            print("======>Current memory:\n %s" % self.memory_thread)
        except:
            print("======>No memory")
        print("failed here")
        response = self.reply(text)
        state = state + [(text, response)]
        print("Outputs:", state)
        return state, state
