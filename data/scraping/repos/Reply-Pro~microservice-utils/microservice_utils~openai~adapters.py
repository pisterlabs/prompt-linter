import typing
from dataclasses import dataclass
from pprint import pprint

import openai
from masked_ai.masker import Masker

MASKER_BOUNDARY = " |-rcxmessageboundary-| "


@dataclass
class OpenAiChatMessage:
    content: str
    role: typing.Literal["assistant", "system", "user"]


@dataclass
class MaskedMessages:
    masker: Masker
    messages: list[dict]


class OpenAiLlm:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        mask_prompts: bool = True,
        max_tokens: int = 2049,
        temperature: int = 0,
    ):
        openai.api_key = api_key
        self._mask_prompts = mask_prompts
        self._max_tokens = max_tokens
        self._model = model
        self._temperature = temperature

    @staticmethod
    def get_masked_chat_messages(
        unmasked_messages: list[OpenAiChatMessage],
    ) -> MaskedMessages:
        masker = Masker(MASKER_BOUNDARY.join([m.content for m in unmasked_messages]))

        messages = []

        for i, message in enumerate(masker.masked_data.split(MASKER_BOUNDARY)):
            messages.append({"content": message, "role": unmasked_messages[i].role})

        return MaskedMessages(masker=masker, messages=messages)

    def generate_chat_response(
        self, messages: list[OpenAiChatMessage]
    ) -> OpenAiChatMessage:
        masked_messages = self.get_masked_chat_messages(messages)

        response = openai.ChatCompletion.create(
            model=self._model,
            messages=masked_messages.messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        unmasked_response = masked_messages.masker.unmask_data(
            response.choices[0]["message"]["content"]
        )

        return OpenAiChatMessage(role="assistant", content=unmasked_response)

    def generate_response(self, prompt: str) -> str:
        masker = Masker(prompt)

        response = openai.Completion.create(
            model=self._model,
            prompt=masker.masked_data,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        return masker.unmask_data(response.choices[0].text)


class FakeOpenAiLlm:
    _default_response = "I am your friendly virtual assistant."

    def __init__(self, predefined_responses: list[str] = None, **kwargs):
        self._predefined_responses = predefined_responses or []

    @staticmethod
    def get_masked_chat_messages(
        unmasked_messages: list[OpenAiChatMessage],
    ) -> MaskedMessages:
        return OpenAiLlm.get_masked_chat_messages(unmasked_messages)

    def _get_response(self) -> str:
        if self._predefined_responses:
            return self._predefined_responses.pop(0)

        return self._default_response

    def generate_chat_response(
        self, messages: list[OpenAiChatMessage]
    ) -> OpenAiChatMessage:
        return OpenAiChatMessage(role="assistant", content=self._get_response())

    def generate_response(self, prompt: str) -> str:
        return self._get_response()


if __name__ == "__main__":
    """Test the OpenAI adapter.
    --
    python microservice_utils/openai/adapters.py --api-key 'sk-fake' chat
    --
    python microservice_utils/openai/adapters.py --api-key 'sk-fake' prompt
    """
    import argparse

    def chat(args):
        print(
            "You are chatting with ChatGPT. Type 'quit' to stop or 'dump' to show all "
            "the messages in the current chat"
        )

        adapter = OpenAiLlm(api_key=args.api_key)
        messages = [
            OpenAiChatMessage(
                content="You are a test virtual assistant that helps engineers verify "
                "that their OpenAI integration is working. You can be as "
                "funny as a comedian when you respond.",
                role="system",
            )
        ]

        while True:
            try:
                # Prompt the user for their input
                user_input = input("You: ")

                # Exit the loop if the user inputs "quit"
                if user_input.lower() == "quit":
                    exit()
                elif user_input.lower() == "dump":
                    print("Currently stored messages:")
                    pprint(messages)
                    continue

                # Add the user input to the list of chat messages
                messages.append(OpenAiChatMessage(content=user_input, role="user"))

                # Generate a response
                response = adapter.generate_chat_response(messages)

                # Add the response to the list of chat messages
                messages.append(response)

                # Print the response
                print("Assistant: ", response.content)
            except KeyboardInterrupt:
                print("Quitting...")
                exit()

    def prompt(args):
        adapter = OpenAiLlm(api_key=args.api_key, model="text-davinci-003")

        print(adapter.generate_response(args.prompt))

    parser = argparse.ArgumentParser(description="Chat with OpenAI's ChatGPT")
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key")
    subparsers = parser.add_subparsers(help="Chat or send a single prompt to OpenAI.")

    # Add a subparser for the chat command
    chat_parser = subparsers.add_parser(
        "chat", help="Start a chat session with OpenAI's ChatGPT"
    )
    chat_parser.set_defaults(func=chat)

    # Add a subparser for the prompt command
    prompt_parser = subparsers.add_parser("prompt", help="Prompt OpenAI")
    prompt_parser.add_argument(
        "prompt", type=str, help="Prompt for generating a response"
    )
    prompt_parser.set_defaults(func=prompt)

    args = parser.parse_args()

    try:
        args.func(args)
    except AttributeError:
        print("Please choose a command.")
