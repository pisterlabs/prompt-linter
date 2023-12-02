import logging
import os
import time
from threading import Thread

import openai
from googletrans import Translator

from katia.message_manager import KatiaProducer
from katia.message_manager.consumer import KatiaConsumer

logger = logging.getLogger("KatiaInterpreter")


class KatiaInterpreter(Thread):
    """
    This is the main interpreter. It will run in a separate thread and will listen to the
    kafka topic to check if there is new messages to interpret.

    Once it interprets a message, it will produce a new kafka message for the speaker.

    It is based on the openai technology, so it is needed to be configured the first
    prompt for it setting its context configured by the client.
    """

    def __init__(self, name: str, owner_uuid: str, adjectives: tuple = ()):
        super().__init__()
        logger.info("Starting interpreter")
        self.language = os.getenv("KATIA_LANGUAGE", "en-US")
        try:
            openai.api_key = os.environ["OPENAI_KEY"]
        except KeyError as ex:
            error_message = (
                "Missing OPENAI_KEY for interpreter. This env value is mandatory"
            )
            logger.error(error_message)
            raise EnvironmentError(error_message) from ex
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.name = name
        self.adjectives = adjectives
        self.messages = [{"role": "system", "content": self.initial_prompt}]

        self.consumer = KatiaConsumer(
            topic=f"user-{owner_uuid}-interpreter",
            group_id=owner_uuid
        )
        self.producer = KatiaProducer(
            topic=f"user-{owner_uuid}-speaker",
            group_id=owner_uuid
        )
        self.active = True
        logger.info("Interpreter started")

    def run(self) -> None:
        self.interpret()

    def interpret_message(self, message: str):
        """
        This method will interpret the message calling openai and getting the response
        from them.
        It will append the message as role use message and the response as assistant
        message.
        :param message:
        :return:
        """
        self.messages.append({"role": "user", "content": message})

        logger.info("Calling openai, please wait")
        start = time.time()
        try:
            response = openai.ChatCompletion.create(
                model=self.model, messages=self.messages
            )
            response_text = response["choices"][0]["message"]["content"]
            logger.info(
                "Response from openai obtained in '%s' seconds",
                round(time.time() - start, 2)
            )
            self.messages.append({"role": "assistant", "content": response_text})
        except Exception as ex:
            self.messages.pop()
            logger.error(
                "Something went wrong doing the interpretation of the message",
                extra={
                    "error": str(ex),
                    "err_message": message
                },
            )
            translator = Translator()
            response_text = translator.translate(
                text=(
                    "sorry, something went wrong. It seems that I can not understand "
                    "what are you saying"
                ),
                dest=self.language.split("-", maxsplit=1)[0],
            ).text
        self.producer.send_message(
            message_data={"source": "interpreter", "message": response_text}
        )
        logger.info("Katia response: '%s'", {response_text})

    def interpret(self):
        """
        Main loop for starting recognize data. It will keep looking at the topic set
        in the consumer, checking if the recognizer sent a message
        :return:
        """
        self.ready_to_interpret()
        while self.active:
            data = self.consumer.get_data()
            if (
                data
                and data.get("source", None) == "recognizer"
                and (message := data.get("message", None))
            ):
                self.interpret_message(message)

    @property
    def initial_prompt(self):
        """
        This property is used to return a string with the initial prompt using the
        parameters set for the interpreter. It will also translate the prompt to the
        user language
        :return:
        """
        initial_text = "You are a "
        conjunction = "and"
        ending_text = f"assistant called {self.name}."
        extra_description = os.getenv("KATIA_EXTRA_DESCRIPTION", "")
        if extra_description:
            ending_text += f" {extra_description}"

        if "en" not in self.language:
            initial_text, conjunction, ending_text = self.translate_initial_prompt(
                initial_text=initial_text,
                conjunction=conjunction,
                ending_text=ending_text,
            )

        if self.adjectives:
            initial_text += ", ".join(self.adjectives[:-1])
            initial_text += f" {conjunction} {self.adjectives[-1]}"

        complete_text = f"{initial_text} {ending_text}"
        return complete_text.strip()

    def translate_initial_prompt(
        self,
        initial_text: str,
        conjunction: str,
        ending_text: str,
    ):
        """
        Method to translate the initial prompt to the user language
        :param initial_text:
        :param conjunction:
        :param ending_text:
        :return:
        """
        language_to_use = self.language.split("-", maxsplit=1)[0]
        translator = Translator()
        initial_text = (
            translator.translate(text=initial_text, dest=language_to_use).text + " "
        )
        conjunction = translator.translate(text=conjunction, dest=language_to_use).text
        ending_text = translator.translate(text=ending_text, dest=language_to_use).text

        return initial_text, conjunction, ending_text

    def ready_to_interpret(self):
        """
        This message will be the firsts message to be sent. You will have to wait until
        the message is sent to start talking with the assistant. This way, we make sure
        that the kafka messages are working
        :return:
        """
        starter_message = "All is ready! I will be your assistant!"
        if "en" not in self.language:
            language_to_use = self.language.split("-", maxsplit=1)[0]
            translator = Translator()
            starter_message = translator.translate(
                text=starter_message, dest=language_to_use
            ).text
        self.producer.send_message(
            message_data={"source": "interpreter", "message": starter_message}
        )

    def deactivate(self):
        """
        Method used to stop the main loop of interpreting
        :return:
        """
        self.active = False
