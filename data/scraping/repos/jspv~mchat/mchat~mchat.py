import os
import asyncio
import argparse

from config import settings

from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.schema.messages import messages_to_dict, messages_from_dict
from langchain.memory import ChatMessageHistory

from retry import retry


from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Label, Markdown, Static
from textual.containers import VerticalScroll, ScrollableContainer
from textual.reactive import Reactive
from textual.css.query import NoMatches
from textual import on, log, work
from textual import events
from textual.message import Message
from textual.worker import Worker
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.align import Align
from rich.pretty import pprint
import pyperclip

from mchat.widgets.ChatTurn import ChatTurn
from mchat.widgets.DebugPane import DebugPane
from mchat.widgets.PromptInput import PromptInput
from mchat.widgets.History import HistoryContainer

from mchat.dalle_image_generator import DallEAPIWrapper

from mchat.Conversation import ConversationRecord, Turn

from textual.containers import Vertical, Horizontal


from typing import Any, Dict, List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage, AIMessage, LLMResult
from langchain.chains.conversation.memory import (
    ConversationSummaryBufferMemory,
    ConversationBufferMemory,
)

DEFAULT_PERSONA_FILE = "mchat/default_personas.json"
EXTRA_PERSONA_FILE = "extra_personas.json"

# Tracing settings for debugging
# os.environ["LANGCHAIN_TRACING"] = "true"
# os.environ["LANGCHAIN_HANDLER"] = "langchain"
# os.environ["LANGCHAIN_SESSION"] = "callback_testing"  # This is session


class StreamTokenCallback(AsyncCallbackHandler):
    """Callback handler that posts new tokens to the chatbox."""

    def __init__(self, app, *args, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    # this method is automatically called by the Langchain callback system when a new
    # token is available
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.app.post_message(
            self.app.AddToChatMessage(role="assistant", message=token)
        )
        await asyncio.sleep(0.05)


class ChatApp(App):
    CSS_PATH = "mchat.tcss"
    BINDINGS = [
        ("ctrl+r", "toggle_dark", "Toggle dark mode"),
        ("ctrl+g", "toggle_debug", "Toggle debug mode"),
    ]

    # Toggles debug pane on/off (default is off)
    _show_debug = Reactive(False)

    # placeholder for the current question
    _current_question = Reactive("")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # parse arguments
        self.parse_args_and_initialize()

        # current active widget for storing a conversation turn
        self.chatbox = None

        # Initialize Conversation record used to store conversation details
        self.record = ConversationRecord()

        # load standard personas
        if os.path.exists(DEFAULT_PERSONA_FILE):
            import json

            with open(DEFAULT_PERSONA_FILE) as f:
                self.personas = json.load(f)
        else:
            raise ValueError("no default_personas.json file found")

        # if there is an EXTRA_PERSONA_FILE, load the personas from there
        if os.path.exists(EXTRA_PERSONA_FILE):
            import json

            with open(EXTRA_PERSONA_FILE) as f:
                extra_personas = json.load(f)
            self.personas.update(extra_personas)

        # load the llm models from settings - name: {key: value,...}
        self.llm_models = {}
        for model_family in settings.llm_model_families:
            for model in getattr(settings, model_family):
                api_key = getattr(settings, f"{model_family}_api_key")
                # see if the model has any parameters
                if getattr(settings, model, None) is not None:
                    self.llm_models[model] = dict(getattr(settings, model))
                    self.llm_models[model]["api_key"] = api_key
                else:
                    self.llm_models[model] = {"api_key": api_key}
        self.available_llm_models = self.llm_models.keys()

        # load the image models from settings - name: {key: value,...}
        self.image_models = {}
        for model_family in settings.image_model_families:
            for model in getattr(settings, model_family):
                api_key = getattr(settings, f"{model_family}_api_key")
                # see if the model has any parameters
                if getattr(settings, model, None) is not None:
                    self.image_models[model] = dict(getattr(settings, model))
                    self.image_models[model]["api_key"] = api_key
                else:
                    self.image_models[model] = {"api_key": api_key}
        self.available_image_models = self.image_models.keys()

        # Initialize the main language model
        self.llm_model_name = settings.default_llm_model
        self.llm_temperature = settings.default_llm_temperature
        self.llm = self._initialize_model(
            self.llm_model_name, [StreamingStdOutCallbackHandler()]
        )

        # Initialize the summary model
        self.summary_model_name = settings.memory_model
        self.summary_temperature = settings.memory_model_temperature
        self.summary_max_tokens = settings.memory_model_max_tokens
        self.summary_llm = self._initialize_model(
            self.summary_model_name,
            override_temperature=self.summary_temperature,
            streaming=False,
        )

        # Initialize the image model
        self.default_image_model = settings.get("default_image_model", None)
        if self.default_image_model is not None:
            self.image_model_name = self.default_image_model
            self.image_model = self._initialize_image_model()

        self.memory = ConversationSummaryBufferMemory(
            llm=self.summary_llm,
            max_token_limit=self.summary_max_tokens,
            return_messages=True,
        )

        self.set_persona(getattr(settings, "default_persona", "default"))

    def _initialize_model(
        self,
        model_name: str,
        callbacks: List[BaseCallbackHandler] = [],
        override_temperature: Optional[float] = None,
        streaming: bool = True,
    ):
        """Initialize the large language model."""
        self.log(f"Initializing llm model {model_name}")

        temperature = override_temperature or self.llm_temperature

        # if the model name starts with ms_, use the AzureChatOpenAI model
        if model_name.startswith("ms_"):
            llm = AzureChatOpenAI(
                deployment_name=self.llm_models[model_name]["deployment"],
                openai_api_base=self.llm_models[model_name]["base_url"],
                openai_api_version=self.llm_models[model_name]["api"],
                openai_api_key=self.llm_models[model_name]["api_key"],
                verbose=False,
                streaming=streaming,
                callbacks=callbacks,
                temperature=temperature,
            )
        elif model_name.startswith("oai_"):
            llm = ChatOpenAI(
                model_name=self.llm_models[model_name]["deployment"],
                openai_api_key=self.llm_models[model_name]["api_key"],
                verbose=False,
                streaming=streaming,
                callbacks=callbacks,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown model type {model_name}")

        return llm

    def _initialize_image_model(self):
        """initialize the image model."""
        image_model = DallEAPIWrapper(
            openai_api_key=self.image_models[self.image_model_name]["api_key"],
            num_images=self.image_models[self.image_model_name]["num_images"],
            size=self.image_models[self.image_model_name]["size"],
            model=self.image_models[self.image_model_name]["model"],
            quality=self.image_models[self.image_model_name]["quality"],
        )
        return image_model

    def _reinitialize_llm_model(self, messages: List[str] = []):
        """re-initialize the language model."""
        self.llm = self._initialize_model(
            self.llm_model_name, [StreamingStdOutCallbackHandler()]
        )

        # if there are messages, we're restring a historical session, create new
        # memory and reinitialize the conversation
        if len(messages) > 0:
            self.memory = ConversationSummaryBufferMemory(
                llm=self.summary_llm,
                max_token_limit=self.summary_max_tokens,
                return_messages=True,
                chat_memory=ChatMessageHistory(messages=messages_from_dict(messages)),
            )
            debug_pane = self.query_one(DebugPane)
            debug_pane.update_entry(
                "history", lambda: self.memory.load_memory_variables({})["history"]
            )
            debug_pane.update_entry(
                "summary_buffer",
                lambda: self.memory.moving_summary_buffer,
            )

        self.conversation = ConversationChain(
            llm=self.llm,
            verbose=False,
            prompt=self.prompt,
            memory=self.memory,
        )

        debug_pane = self.query_one(DebugPane)
        debug_pane.update_status()

    def parse_args_and_initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-v", "--verbose", help="Increase verbosity", action="store_true"
        )
        args, unknown = parser.parse_known_args()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Horizontal():
            yield DebugPane()
            yield HistoryContainer(
                new_label="New Session", summary_model=self.summary_llm
            )
            with Vertical():
                self.chat_container = VerticalScroll(id="chat-container")
                yield self.chat_container
                yield PromptInput()
        yield Footer()

    def on_mount(self) -> None:
        self.title = "mchat - Multi-Model Chatbot"

        # set focus to the input box
        input = self.query_one(PromptInput)
        input.focus()

    def on_ready(self) -> None:
        """Called  when the DOM is ready."""
        input = self.query_one(PromptInput)
        input.focus()

        debug_pane = self.query_one(DebugPane)
        debug_pane.add_entry("model", "Model", lambda: self.llm_model_name)
        debug_pane.add_entry("temp", "Temperature", lambda: self.llm_temperature)
        debug_pane.add_entry("persona", "Persona", lambda: self.current_persona)
        debug_pane.add_entry("question", "Question", lambda: self._current_question)
        debug_pane.add_entry("prompt", "Prompt", lambda: self.app.conversation.prompt)
        debug_pane.add_entry(
            "history",
            "History",
            lambda: self.memory.load_memory_variables({})["history"],
        )
        debug_pane.add_entry("memref", "Memory Reference", lambda: self.memory)
        debug_pane.add_entry(
            "summary_buffer",
            "Summary Buffer",
            lambda: self.memory.moving_summary_buffer,
        )

    @on(ChatTurn.ChatTurnClicked)
    def click_chat_turn(self, event: events) -> None:
        chatturn = event.widget

        # copy contents of chatbox to clipboard
        pyperclip.copy(chatturn.message)

    @on(PromptInput.Submitted)
    def submit_question(self, event: events) -> None:
        input = self.query_one(PromptInput)
        self.post_message(self.AddToChatMessage(role="user", message=event.value))

        # clear the input box
        input.value = ""

        # ask_question is a work function, so it will be run in a separate thread

        self.ask_question(event.value)
        self.post_message(self.EndChatTurn(role="user"))

    def on_key(self, event: events.Key) -> None:
        """Write Key events to log."""
        pass
        # text_log = self.query_one(TextLog)
        # text_log.write(event)
        # self.post_message(self.AddToChatMessperage("keypress"))

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark

    def action_toggle_debug(self) -> None:
        """An action to toggle debug mode."""
        self._show_debug = not self._show_debug

    def count_tokens(self, chain, query):
        with get_openai_callback() as cb:
            result = chain.run(query)
        print(result)
        print(f"Spent a total of {cb.total_tokens} tokens\n")
        return result

    def watch__show_debug(self, show_debug: bool) -> None:
        """When __show_debug changes, toggle the class the debug widget."""
        self.app.set_class(show_debug, "-show-debug")

    # there is a bug in actually using templates with the memory object, so we
    # build the prompt template manually

    # build the prompt template; note: the MessagesPlaceholder is required
    # to be able to access the history of messages, its variable "history" will be
    # replaced with the history of messages by the conversation chain as provided by
    # the memory object.
    def build_prompt_template(self, persona):
        base = []

        # Initial system message
        if len(self.personas[persona]["description"]) > 0:
            base.append(
                SystemMessagePromptTemplate.from_template(
                    self.personas[persona]["description"]
                )
            )
        # Extra system messages
        for extra in self.personas[persona]["extra_context"]:
            if extra[0] == "ai":
                base.append(AIMessagePromptTemplate.from_template(extra[1]))
            elif extra[0] == "human":
                base.append(HumanMessagePromptTemplate.from_template(extra[1]))
            elif extra[0] == "system":
                base.append(SystemMessagePromptTemplate.from_template(extra[1]))
            else:
                raise ValueError(f"Unknown extra context type {extra[0]}")

        # History Placeholder
        base.append(MessagesPlaceholder(variable_name="history"))

        # Human message
        base.append(HumanMessagePromptTemplate.from_template("{input}"))

        self.log.debug(f"Prompt template is: {base}")

        return ChatPromptTemplate.from_messages(base)

    def set_persona(self, persona: str):
        """Set the persona and reinitialize the conversation chain."""

        self.current_persona = persona
        if persona not in self.personas:
            raise ValueError(f"Persona '{persona}' not found")
        # have to rebuild prompt and chain due to
        # https://github.com/hwchase17/langchain/issues/1800 - can't use templates
        self.prompt = self.build_prompt_template(persona=persona)

        self.conversation = ConversationChain(
            llm=self.llm,
            verbose=False,
            prompt=self.prompt,
            memory=self.memory,
        )

    # Add addtional retry logic to the ask_question function
    @retry(tries=3, delay=1)
    async def _ask_question_to_llm(
        self, question: str, callbacks: List[BaseCallbackHandler]
    ):
        await self.conversation.arun(question, callbacks=callbacks)

    @work(exclusive=True)
    async def ask_question(self, question: str):
        """Ask a question to the AI and return the response.  Textual work function."""

        # scroll the chat container to the bottom
        self.scroll_to_end()

        # if the question is 'new' or 'new session" start a new session
        if question.lower() == "new" or question == "new session":
            # if the session we're in is empty, don't start a new session
            if len(self.record.turns) == 0:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant",
                        message="You're already in a new session",
                    )
                )
                self.post_message(self.EndChatTurn(role="meta"))
                return

            # clear the chatboxes from the chat container
            self.chat_container.remove_children()

            # start a new history session
            history = self.query_one(HistoryContainer)
            await history.new_session()
            self.memory.clear()
            self.record = ConversationRecord()
            self.set_persona(getattr(settings, "default_persona", "default"))
            self.post_message(self.EndChatTurn(role="meta"))
            return

        # if the question is either 'persona', or 'personas' show the available personas
        if question == "personas" or question == "persona":
            self.post_message(
                self.AddToChatMessage(role="assistant", message="Available Personas:\n")
            )
            for persona in self.personas:
                self.post_message(
                    self.AddToChatMessage(role="assistant", message=f" - {persona}\n")
                )
            self.post_message(self.EndChatTurn(role="meta"))
            return

        if question.startswith("persona"):
            # load the new persona
            persona = question.split(maxsplit=1)[1].strip()
            if persona not in self.personas:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant", message=f"Persona '{persona}' not found"
                    )
                )
                return
            self.post_message(
                self.AddToChatMessage(
                    role="assistant", message=f"Setting persona to '{persona}'"
                )
            )
            self._current_question = ""
            self.set_persona(persona=persona)

            self.post_message(self.EndChatTurn(role="assistant"))
            return

        # if the question is 'models', or 'model', show available models
        if question == "models" or question == "model":
            self.post_message(
                self.AddToChatMessage(role="assistant", message="Available Models:\n")
            )
            for model in self.available_llm_models:
                self.post_message(
                    self.AddToChatMessage(role="assistant", message=f" - {model}\n")
                )
            self._current_question = ""
            self.post_message(self.EndChatTurn(role="meta"))
            return

        # if the question starts with 'model', set the model
        if question.startswith("model"):
            # get the model
            llm_model_name = question.split(maxsplit=1)[1].strip()

            # check to see if the model is valid
            if llm_model_name not in self.available_llm_models:
                self.post_message(
                    self.AddToChatMessage(
                        role="assistant",
                        message=f"Model '{self.llm_model_name}' not found",
                    )
                )
                self._current_question = ""
                self.post_message(self.EndChatTurn(role="meta"))
                return

            self.llm_model_name = llm_model_name

            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message=f"Model set to {self.llm_model_name}",
                )
            )
            self._current_question = question
            self._reinitialize_llm_model()
            self.post_message(self.EndChatTurn(role="assistant"))
            return

        # if the question starts with 'summary', summarize the conversation
        if question.startswith("summary"):
            # get the summary
            summary = str(self.memory.load_memory_variables({}))
            # post the summary
            self._current_question = ""
            self.post_message(self.AddToChatMessage(role="assistant", message=summary))
            self.post_message(self.EndChatTurn(role="meta"))
            return

        # if the question starts with 'temperature', set the temperature
        if question.startswith("temperature"):
            # get the temperature
            self.llm_temperature = float(question.split(maxsplit=1)[1].strip())
            # post the summary
            self.post_message(
                self.AddToChatMessage(
                    role="assistant",
                    message=f"Temperature set to {self.llm_temperature}",
                )
            )
            self._current_question = question
            self._reinitialize_llm_model()
            self.post_message(self.EndChatTurn(role="assistant"))
            return

        # hack - testing
        # if question starts with 'dall-e ' pass to Dall-e
        if question.startswith("dall-e "):
            question = question[7:]
            self.post_message(
                self.AddToChatMessage(role="assistant", message="Generating...")
            )
            self._current_question = question
            self.post_message(self.EndChatTurn(role="meta"))
            out = await self.image_model.arun(question)
            self.post_message(self.AddToChatMessage(role="assistant", message=out))
            # out = "[image](" + out + ")"
            # self.post_message(self.AddToChatMessage(role="assistant", message=out))
            self.post_message(self.EndChatTurn(role="assistant"))
            return

        self._current_question = question

        # ask the question and wait for a response
        await self._ask_question_to_llm(question, [StreamTokenCallback(self)])

        # Done with response; clear the chatbox
        self.scroll_to_end()
        self.post_message(self.EndChatTurn(role="assistant"))

    def run(self, *args, **kwargs):
        """Run the app."""

        try:
            super().run(*args, **kwargs)
        except asyncio.exceptions.CancelledError:
            self.log.debug("Markdown crashed\n{}", self.chatbox.markdown)

    def scroll_to_end(self) -> None:
        if self.chat_container is not None:
            self.chat_container.refresh()
            self.chat_container.scroll_end(animate=False)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when the worker state changes."""
        self.log(event)

    # Built-in Utility functions automatically called by langchain callbacks

    async def new_token_post(self, token: str) -> None:
        """Post a new token to the chatbox."""
        self.post_message(self.AddToChatMessage(role="assistant", message=token))

    # Custom Message Handlers - these are in two parts:
    # 1. A Message class that is used to send the message and embed data
    # 2. A function that is decorated with @on(MessageClass) that will be called when
    #    the message is received

    class AddToChatMessage(Message):
        def __init__(self, role: str, message: str) -> None:
            assert role in ["user", "assistant"]
            self.role = role
            self.message = message
            super().__init__()

    @on(AddToChatMessage)
    async def add_to_chat_message(self, chat_token: AddToChatMessage) -> None:
        chunk = chat_token.message
        role = chat_token.role

        # Create a ChatTurn widget if we don't have one and mount it in the container
        # make sure to scroll to the bottom
        if self.chatbox is None:
            self.chatbox = ChatTurn(role=role)
            await self.chat_container.mount(self.chatbox)
            self.chat_container.scroll_end(animate=False)

        await self.chatbox.append_chunk(chunk)

        self.app.log.debug('message is: "{}"'.format(self.chatbox.message))
        self.app.log.debug('message markdown is: "{}"'.format(self.chatbox.markdown))

        # if we're not near the bottom, scroll to the bottom
        scroll_y = self.chat_container.scroll_y
        max_scroll_y = self.chat_container.max_scroll_y
        if scroll_y in range(max_scroll_y - 3, max_scroll_y + 1):
            self.chat_container.scroll_end(animate=False)

    class EndChatTurn(Message):
        def __init__(self, role: str) -> None:
            assert role in ["user", "assistant", "meta"]
            self.role = role
            super().__init__()

    @on(EndChatTurn)
    async def end_chat_turn(self, event: EndChatTurn) -> None:
        """Called when the worker state changes."""
        # If we hae a    response, add the turn to the conversation record
        if event.role == "assistant":
            self.record.add_turn(
                persona=self.current_persona,
                prompt=self._current_question,
                response=self.chatbox.message,
                summary=self.memory.moving_summary_buffer,
                model=self.llm_model_name,
                temperature=self.llm_temperature,
                memory_messages=messages_to_dict(self.memory.chat_memory.messages),
            )

            history = self.query_one(HistoryContainer)
            await history.update_conversation(self.record)

        # Update debug pane
        debug_pane = self.query_one(DebugPane)
        debug_pane.update_status()

        # if we have a chatbox, close it.
        self.chatbox = None

    @on(HistoryContainer.HistorySessionClicked)
    def _on_history_session_clicked(
        self, event: HistoryContainer.HistorySessionClicked
    ) -> None:
        """Restore the previous session"""

        event.stop()

        # if the record is the same as the current record, do nothing
        if self.record == event.record:
            return

        self.record = event.record

        # if the record is none, it is an empty session, so clear the chatbox
        if event.record is None:
            self._current_question = ""
            self.memory.clear()
            self.chat_container.remove_children()
            return

        # if there are no turns in the record, it's a new session
        if len(self.record.turns) == 0:
            self._current_question = ""
            self.memory.clear()
        else:
            self.current_persona = self.record.turns[-1].persona
            self.llm_model_name = self.record.turns[-1].model
            self.llm_temperature = self.record.turns[-1].temperature
            self._current_question = self.record.turns[-1].prompt
            self._reinitialize_llm_model(messages=self.record.turns[-1].memory_messages)

        # clear the chatboxes from the chat container
        self.chat_container.remove_children()

        # load the chat history from the record
        for turn in self.record.turns:
            self.post_message(self.AddToChatMessage(role="user", message=turn.prompt))
            self.post_message(self.EndChatTurn(role="meta"))
            self.post_message(
                self.AddToChatMessage(role="assistant", message=turn.response)
            )
            self.post_message(self.EndChatTurn(role="meta"))


def main():
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
