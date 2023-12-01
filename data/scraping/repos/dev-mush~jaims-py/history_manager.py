from io import BytesIO
from typing import List, Optional
from jaims.openai_wrappers import (
    estimate_token_count,
    JAImsOptions,
    JAImsOpenaiKWArgs,
)

from jaims.exceptions import JAImsTokensLimitExceeded
import json

from jaims.function_handler import parse_function_wrappers_to_tools
from math import ceil
import base64
from PIL import Image


class HistoryManager:
    """
    Manages the history of messages sent to OpenAI.
    The history manager stores the messages in memory and, when the agent is set to optimize the context,
    it optimizes the history to fit the max tokens supported by the current llm model.

    Parameters
    ----------
        history : list (optional)
            the history of messages to be sent to openai
    """

    def __init__(
        self,
        history: Optional[List] = None,
    ):
        """
        Returns a new HistoryManager instance.

        Parameters
        ----------
            history : list (optional)
                the initial history of messages to be sent to openai
        """
        self.__history = history or []

    def add_messages(self, messages: List):
        """
        Pushes new messages in the history.

        Parameters
        ----------
            message : str
                the message to be added
        """

        if not all(isinstance(message, dict) for message in messages):
            raise TypeError(
                "All messages must be dicts, conforming to OpenAI API specification."
            )

        # this workaround is necessary because openai made a mess
        # with the new return types of the openai api, that were just a plain dictionary before and
        # now are a thousand classes often identical to each other.
        # I'm adding content because when I do the model_dump() of the model, None values are skipped
        # but "content" must be passed to none otherwise the api breaks.
        for message in messages:
            if "content" not in message:
                message["content"] = None

        self.__history.extend(messages)

    def get_messages_for_current_run(
        self,
        options: JAImsOptions,
        openai_kwargs: JAImsOpenaiKWArgs,
    ) -> List:
        """
        Returns the messages to be sent to openai for the current run.

        Parameters
        ----------
            options : JAImsOptions
                the options for the current run
            openai_kwargs : JAImsOpenaiKWArgs
                the openai kwargs for the current run

        Raises
        ------
            TokensLimitExceeded
                if the max tokens to be used exceed the max tokens supported by the current llm model

        Developer Notes
        ---------------

        # Optimization Feature
        This is roughly how the optimization feature works right now, plus some notes on how I intend to improve it.
        The tokens for the messages to send to openai are calculated by composing the mandatory_context, the chat history managed by this class
        and the functions in a variable named compound_history.
        They are parsed to a json string and tiktoken evaluates the token consumption based on the current llm model setting.

        Right now I'm parsing the whole compound_history, including the functions, to json and calculating the tokens based on the json string,
        but I'm not sure how accurate this is.
        On rough estimates made with the tokenizer from openai they seem to be pretty accurate, but I need to investigate more.

        A high level overview of the optimization loop is the following:

        the max_tokens to be used are the max tokens supported by the current llm model, minus the tokens to leave out for the response
        from openai passed in agent_max_tokens (passed with a default value) .

        The compound_history tokens are calculated, and while they exceed the context max_tokens:
        1. the first (oldest) message from the chat history between the user and the angent is popped
        2. the compound_history tokens are recalculated
        3. if the compound_history tokens are still above the context max_tokens, the process is repeated from step 1
            3.1. if the chat history between user and agent remains empty for some reason
                (this could happen for instance if functions and mandatory context are way too big),
                an exception is raised. I think it aids development but have to think about it.
        4. if the compound_history tokens are below the context max_tokens, the mandatory_context + the optimized history are returned.

        TODO:
        - optimize functions messages in the history

        MAYBE TODO:
        - it would be nice to have an auto-scale up of the context, for instance passing from the gpt-3.5-turbo 4k to the 16k model.
        """

        if not options or not openai_kwargs:
            raise ValueError("options and openai_kwargs must be provided.")

        # Copying the whole history to avoid altering the original one
        history_buffer = self.__history.copy()

        # If last_n_turns is set, only keep the last n messages
        if options.message_history_size is not None:
            history_buffer = history_buffer[-options.message_history_size :]

        # create the compound history with the mandatory context
        # the actual chat history and the functions to calculate the tokens
        json_functions = parse_function_wrappers_to_tools(openai_kwargs.tools or [])
        leading_prompts = options.leading_prompts or []
        trailing_prompts = options.trailing_prompts or []
        compound_history = (
            leading_prompts + history_buffer + (json_functions) + trailing_prompts
        )

        # the max tokens to be used are the max tokens supported by the current
        # openai model minus the tokens to leave out for the response from openai
        context_max_tokens = openai_kwargs.model.max_tokens - openai_kwargs.max_tokens

        # calculate the tokens for the compound history
        messages_tokens = self.__tokens_from_messages(
            compound_history, openai_kwargs.model
        )

        if options.optimize_context:
            while messages_tokens > context_max_tokens:
                if not history_buffer:
                    raise JAImsTokensLimitExceeded(
                        openai_kwargs.model.max_tokens,
                        messages_tokens,
                        openai_kwargs.max_tokens,
                        has_optimized=True,
                    )

                # Popping the first (oldest) message from the chat history between the user and agent
                history_buffer.pop(0)

                # Recalculating the tokens for the compound history
                messages_tokens = self.__tokens_from_messages(
                    leading_prompts
                    + history_buffer
                    + json_functions
                    + trailing_prompts,
                    openai_kwargs.model,
                )
        elif messages_tokens > context_max_tokens:
            raise JAImsTokensLimitExceeded(
                openai_kwargs.model.max_tokens,
                messages_tokens,
                openai_kwargs.max_tokens,
                has_optimized=False,
            )

        llm_messages = leading_prompts + history_buffer + trailing_prompts

        return llm_messages

    def clear_history(self):
        """
        Clears the history.
        """
        self.__history = []

    def get_history(self):
        """
        Returns entire history.
        """
        return self.__history

    def __tokens_from_messages(self, messages: List, model):
        """Returns the number of tokens used by a list of messages."""

        images = []
        parsed = []
        for message in messages:
            message_copy = message.copy()

            if isinstance(message.get("content", None), list):
                filtered_content = []
                for item in message["content"]:
                    if (
                        isinstance(item, dict)
                        and item.get("image_url", None)
                        and item["image_url"]["url"].startswith(
                            "data:image/jpeg;base64,"
                        )
                    ):
                        images.append(
                            item["image_url"]["url"].replace(
                                "data:image/jpeg;base64,", ""
                            )
                        )
                    else:
                        filtered_content.append(item)
                message_copy["content"] = filtered_content
            parsed.append(message_copy)

        image_tokens = 0
        for image in images:
            width, height = self.__get_image_size_from_base64(image)
            image_tokens += self.__count_image_tokens(width, height)

        return estimate_token_count(json.dumps(parsed), model) + image_tokens

    def __get_image_size_from_base64(self, base64_string):
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))

        return image.size

    def __count_image_tokens(self, width: int, height: int):
        h = ceil(height / 512)
        w = ceil(width / 512)
        n = w * h
        total = 85 + 170 * n
        return total
