from textual.widget import Widget
from textual.widgets import Static
from textual.message import Message
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual import events
from textual.events import Click
from textual import on
from dataclasses import dataclass
from rich.console import RenderableType
from rich.text import Text

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel

from retry.api import retry

from mchat.Conversation import ConversationRecord

import apsw.bestpractice


"""
Design:
HistoryContainer Widget is a vertical scroll container that displays a list of Chat
History sesisons represented by HistorySessionBox widgets.  Each HistorySessionBox
widget represents a single ConversationRecord.  The HistorySessionBox widget is
composed of a summary box, a copy button, and delete button.
"""


class HistorySessionBox(Widget, can_focus=True):
    """A Widget that displays a single Chat History session."""

    def __init__(
        self,
        record: ConversationRecord | None = None,
        new_label: str = "",
        label: str = "",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.new_label = new_label
        self.label = label
        self.record = record

    def compose(self) -> None:
        with Vertical():
            with Horizontal():
                yield Static(
                    Text("..."),
                    id="history-session-box-label",
                    classes="history-session-box-top",
                )
                yield Static("", id="history-session-box-spacer")
                self.copy_button = CopyButton(id="history-session-box-copy")
                yield self.copy_button
                yield DeleteButton(id="history-session-box-delete")
            self.summary_box = SummaryBox()
            if self.record is None or len(self.record.turns) == 0:
                self.summary_box.update(self.new_label)
                self.copy_button.visible = False
            else:
                if len(self.label) > 0:
                    self.summary_box.update(self.label)
                else:
                    self.summary_box.update(self.record.turns[-1].prompt)
            yield self.summary_box

    async def update_box(
        self, record: ConversationRecord, summary: str | RenderableType = ""
    ) -> None:
        """Update the HistorySessionBox with a new ConversationRecord."""

        self.record = record

        # enable the copy button if it is disabled
        self.copy_button.visible = True

        # update the summary box with the new summary
        if len(summary) > 0:
            summary = Text(summary) if isinstance(summary, str) else summary
            self.summary_box.update(summary)
            self.record.summary = str(summary)
            return

        if len(record.turns) == 0:
            return

        # no summary was provided, so update the summary box with the last prompt
        self.summary_box.update(Text(record.turns[-1].prompt))
        self.record.summary = str(record.turns[-1].prompt)

    def on_click(self, event: Click) -> None:
        event.stop()
        self.post_message(self.Clicked(self, "load"))

    @dataclass
    class Clicked(Message):
        clicked_box: Widget
        action: str

    class ChildClicked(Clicked):
        pass

    @on(ChildClicked)
    def update_click_and_bubble_up(self, event: events) -> None:
        """update the widget with self and bubble up the DOM"""
        event.stop()
        self.post_message(HistorySessionBox.Clicked(self, event.action))


class DeleteButton(Static):
    """A button that deletes the session when clicked."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(":x:", *args, **kwargs)

    def on_click(self, event: Click) -> None:
        event.stop()
        self.app.log.debug("Delete button clicked")
        self.post_message(HistorySessionBox.ChildClicked(self, "delete"))


class CopyButton(Static):
    """A button that deletes the session when clicked."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(":clipboard:", *args, **kwargs)

    def on_click(self, event: Click) -> None:
        event.stop()
        self.app.log.debug("Copy button clicked")
        self.post_message(HistorySessionBox.ChildClicked(self, "copy"))


class SummaryBox(Static):
    """A summary of the current chat."""

    pass


class HistoryContainer(VerticalScroll):
    """A vertical scroll container that displays a list of Chat History sessions."""

    prompt_template = """
    Here is list of chat submissions in the form of 'user: message'. Give me no more 
    than 10 words which will be used to remind the user of the conversation.  Your reply
    should be no more than 10 words and at most 66 total characters.
    Chat submissions: {conversation}
    """

    def __init__(
        self, summary_model: BaseChatModel, new_label: str = "", *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.new_label = new_label

        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=["conversation"]
        )
        self.llm_chain = LLMChain(prompt=self.prompt, llm=summary_model)

        # Open or create the database
        self.conneciton = self._initialize_db()

        # If there are records in the database, load them
        cursor = self.conneciton.cursor()
        rows = cursor.execute("SELECT id FROM Conversations").fetchall()
        if len(rows) > 0:
            for row in rows:
                record = self._read_conversation_from_db(self.conneciton, row[0])
                self._add_previous_session(record)

    def compose(self) -> None:
        self.active_session = HistorySessionBox(new_label=self.new_label)
        self.active_session.add_class("-active")
        yield self.active_session

    # Initialize the database and return a connection object
    def _initialize_db(self) -> apsw.Connection:
        # SQLite3 stuff
        apsw.bestpractice.apply(apsw.bestpractice.recommended)

        # Default will create the database if it doesn't exist
        connection = apsw.Connection("db.db")
        cursor = connection.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS Conversations (id TEXT PRIMARY KEY, data TEXT)"
        )
        return connection

    def _write_conversation_to_db(self, conn, conversation):
        cursor = conn.cursor()
        serialized_conversation = conversation.to_json()
        cursor.execute(
            "INSERT OR REPLACE INTO Conversations (id, data) VALUES (?, ?)",
            (
                conversation.id,
                serialized_conversation,
            ),
        )

    def _read_conversation_from_db(self, conn, id) -> ConversationRecord:
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT data FROM Conversations WHERE id=?", (id,)
        ).fetchone()
        conversation = ConversationRecord.from_json(row[0])
        return conversation

    def _delete_conversation_from_db(self, conn, record: ConversationRecord):
        cursor = conn.cursor()
        id = record.id
        cursor.execute("DELETE FROM Conversations WHERE id=?", (id,))

    @(retry(tries=3, delay=1))
    async def get_summary_blurb(self, conversation: str) -> str:
        """Summarize a conversation using the LLMChain model."""
        summary = await self.llm_chain.arun(conversation)
        return summary

    def _add_previous_session(self, record: ConversationRecord) -> None:
        """Add a new session to the HistoryContainer."""
        self.active_session = HistorySessionBox(
            record=record, new_label=self.new_label, label=record.summary
        )
        self.mount(self.active_session)

    async def _add_session(
        self, record: ConversationRecord | None = None, label: str = ""
    ) -> None:
        """Add a new session to the HistoryContainer and set it as active"""
        label = label if len(label) > 0 else ""

        # Create a new session
        self.active_session = HistorySessionBox(
            record=record, new_label=self.new_label, label=label
        )

        # Activate the new session and deactivate all others
        self.active_session.add_class("-active")
        # remove the .-active class from all other boxes
        for child in self.children:
            child.remove_class("-active")
        await self.mount(self.active_session)

    async def update_conversation(self, record: ConversationRecord):
        # Format the last 10 turns of the conversation into a single string
        # to be summarized.  If there are fewer than five turns, summarize
        # the whole conversation.

        # update the database
        self._write_conversation_to_db(self.conneciton, record)

        turns = record.turns
        if len(turns) > 10:
            turns = turns[-10:]
        conversation = "\n".join([f"user:{turn.prompt}" for turn in turns])

        # Summarize the conversation
        summary = await self.get_summary_blurb(conversation)
        await self.active_session.update_box(record, summary)

    async def new_session(self) -> None:
        await self._add_session()

    @property
    def session_count(self) -> int:
        # query the dom for all HistorySessionBox widgets
        sessions = self.query(HistorySessionBox)
        return len(sessions)

    @on(HistorySessionBox.Clicked)
    async def history_session_box_clicked(self, event: events) -> None:
        assert event.action in ["load", "delete", "copy"]
        if event.action == "load":
            # set the clicked box to current and set the .-active class
            self.active_session = event.clicked_box
            self.active_session.add_class("-active")
            # remove the .-active class from all other boxes
            for child in self.children:
                if child != self.active_session:
                    child.remove_class("-active")
            self.post_message(
                HistoryContainer.HistorySessionClicked(self.active_session.record)
            )
            return
        if event.action == "delete":
            # if the clicked box is the only box, reset the current box
            if self.session_count == 1:
                await event.clicked_box.remove()
                await self.new_session()
                self.active_session.record = ConversationRecord()
                # let the app know so it can clear the chat pane
                self.post_message(
                    HistoryContainer.HistorySessionClicked(self.active_session.record)
                )
            elif event.clicked_box == self.active_session:
                # if the clicked box is the current session, set the current session
                # to the previous session or next session if there is no previous
                # session
                current = list(self.query(HistorySessionBox)).index(self.active_session)
                if current == 0:
                    # if the current session is the first session, set the second
                    # session to current
                    self.active_session = list(self.query(HistorySessionBox))[1]
                else:
                    # if the current session is not the first session, set the
                    # previous session to current
                    self.active_session = list(self.query(HistorySessionBox))[
                        current - 1
                    ]

                # remove the clicked session
                # remove the session from the database
                if event.clicked_box.record is not None:
                    self._delete_conversation_from_db(
                        self.conneciton, event.clicked_box.record
                    )
                await event.clicked_box.remove()
                # set the .-active class
                self.active_session.add_class("-active")
                # let the app know so it can update the chat pane
                self.post_message(
                    HistoryContainer.HistorySessionClicked(self.active_session.record)
                )
                return
            else:
                # if the clicked box is not the current session, just remove it
                # remove the session from the database
                self._delete_conversation_from_db(
                    self.conneciton, event.clicked_box.record
                )
                await event.clicked_box.remove()
                return
        if event.action == "copy":
            new_record = event.clicked_box.record.copy()
            await self._add_session(
                new_record,
                label=event.clicked_box.summary_box.renderable,
            )
            self.active_session.record = new_record
            self.post_message(
                HistoryContainer.HistorySessionClicked(self.active_session.record)
            )

    class HistorySessionClicked(Message):
        def __init__(self, record: ConversationRecord) -> None:
            self.record = record
            super().__init__()
