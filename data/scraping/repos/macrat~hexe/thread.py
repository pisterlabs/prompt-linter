import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Self
from zoneinfo import ZoneInfo

import openai

import event
from auth import User
from coderunner import CodeRunner
from history import HistoryDB
from note import Note, NoteDB

TERMS = {
    "a day": 1,
    "a month": 30,
    "a year": 365,
    "forever": 1000 * 365,
}

EventHandler = Callable[[event.Event], Awaitable[None]]


class ThreadManager:
    __singleton: Self | None = None

    def __new__(cls, *args, **kwargs) -> "ThreadManager":
        if cls.__singleton is None:
            cls.__singleton = super().__new__(cls)

        return cls.__singleton

    def __init__(self, history: HistoryDB, notes: NoteDB) -> None:
        super().__init__()

        self.threads: dict[str, Thread] = {}
        self.history = history
        self.notes = notes

    def get(self, user: User) -> "Thread":
        if user.id not in self.threads:
            self.threads[user.id] = Thread(
                user,
                self.history,
                self.notes,
            )
        return self.threads[user.id]

    async def shutdown(self) -> None:
        await asyncio.gather(*[thread.shutdown() for thread in self.threads.values()])


class Thread:
    user: User
    history: HistoryDB
    notes: NoteDB
    event_handlers: list[EventHandler]
    timezone: ZoneInfo
    runners: dict[str, CodeRunner]

    def __init__(
        self,
        user: User,
        history: HistoryDB,
        notes: NoteDB,
        timezone: ZoneInfo = ZoneInfo("UTC"),
    ) -> None:
        self.user = user
        self.history = history
        self.notes = notes
        self.event_handlers = []
        self.timezone = timezone
        self.runners = {}
        with open("prompt.txt", "r") as f:
            self.prompt = f.read()

    async def shutdown(self) -> None:
        for runner in self.runners.values():
            await runner.shutdown()

    async def __event(self, ev: event.Event) -> None:
        if not ev.delta:
            self.history.put(self.user.id, ev)

        await asyncio.gather(*[handler(ev) for handler in self.event_handlers])

    def subscribe(self, handler: EventHandler) -> None:
        self.event_handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        self.event_handlers.remove(handler)

    def stream(self) -> "EventReader":
        return EventReader(self)

    async def send_message(self, message: str) -> None:
        """Send a message to the thread.

        :param message: User message to send.
        """

        user_ev = event.User(content=message)

        await self.__event(user_ev)

        await self.__event(
            event.Status(
                source=user_ev.id,
                generating=True,
                created_at=user_ev.created_at,
            )
        )

        try:
            await self.__invoke(user_ev.id)
        except Exception as err:
            err_ev = event.Error(
                content=f"Failed to invoke AI.\n> {err}",
                source=user_ev.id,
            )
            await self.__event(err_ev)
            raise Exception(
                "Failed to invoke AI."
            ) from err  # TODO: DEBUG: remove this line
        finally:
            await self.__event(
                event.Status(
                    source=user_ev.id,
                    generating=False,
                )
            )

    async def __invoke(self, source: uuid.UUID) -> None:
        """Invoke AI using messages so far."""

        assi_ev = event.Assistant(content="", source=source)

        last_user_msg = self.history.last_user_message(self.user.id)
        notes = []
        if last_user_msg is not None:
            notes = self.notes.query(self.user.id, last_user_msg)
            n_tokens = 0
            for i, note in enumerate(notes):
                n_tokens += note.n_tokens
                if n_tokens > 1024:
                    notes = notes[:i]
                    break

        system_prompt = "\n".join(
            [
                self.prompt,
                "",
                f"User name: {self.user.name}",
                f"Current datetime: {datetime.now(self.timezone).isoformat()}",
                "",
                "==========",
                "Notes:",
                (
                    "\n---\n".join(
                        [
                            f"{note.content} ({note.created_at.astimezone(self.timezone).isoformat()})"
                            for note in notes
                        ]
                    )
                    if len(notes) > 0
                    else "(Notes related to the topic are not found)"
                ),
            ]
        )

        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                *event.as_messages(
                    [x.event for x in self.history.load(self.user.id, 2 * 1024)]
                ),
            ],
            functions=[
                {
                    "name": "save_notes",
                    "description": "Save notes to remember it later.",
                    "parameters": {
                        "type": "object",
                        "required": ["notes"],
                        "properties": {
                            "notes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["content", "available_term"],
                                    "properties": {
                                        "content": {
                                            "desctiption": (
                                                "The content to save. Follow 5W1H"
                                                " method to write each note."
                                            ),
                                            "type": "string",
                                        },
                                        "available_term": {
                                            "description": (
                                                "How long the information is meaningful"
                                                " and useful."
                                            ),
                                            "type": "string",
                                            "enum": list(TERMS.keys()),
                                        },
                                    },
                                },
                                "minItems": 1,
                            },
                        },
                    },
                },
                {
                    "name": "search_notes",
                    "description": (
                        "Search notes that you saved. The result include IDs, created"
                        " timestamps, and note contents."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {
                                "type": "string",
                            },
                        },
                    },
                },
                {
                    "name": "delete_notes",
                    "description": "Delete notes that you saved.",
                    "parameters": {
                        "type": "object",
                        "required": ["ids"],
                        "properties": {
                            "ids": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                },
                                "minItems": 1,
                            },
                        },
                    },
                },
                {
                    "name": "run_code",
                    "description": (
                        "Run code in a Jupyter environment, and returns the output and"
                        " the result. To install packages, you can use `!pip install"
                        " <package>` for Python, and `apt-get install <package>` for"
                        " Bash."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": ["language", "code"],
                        "properties": {
                            "language": {
                                "type": "string",
                                "enum": ["python", "bash"],
                            },
                            "code": {
                                "type": "string",
                            },
                        },
                    },
                },
                # {
                #    "name": "generate_image",
                # },
            ],
            user="hexe/" + self.user.id,
            stream=True,
        )

        func_ev: event.FunctionCall | None = None

        async for chunk in completion:
            delta = chunk.choices[0].delta

            if "content" in delta and delta.content is not None:
                assi_ev.content += delta.content

                await self.__event(
                    event.Assistant(
                        id=assi_ev.id,
                        content=delta.content,
                        created_at=assi_ev.created_at,
                        source=source,
                        delta=True,
                    )
                )

            if "function_call" in delta:
                if func_ev is None:
                    func_ev = event.FunctionCall(
                        id=assi_ev.id,
                        name=delta.function_call.name,
                        arguments=delta.function_call.arguments,
                        source=source if assi_ev.content == "" else assi_ev.id,
                    )
                else:
                    func_ev.arguments += delta.function_call.arguments

                await self.__event(
                    event.FunctionCall(
                        id=func_ev.id,
                        name=func_ev.name,
                        arguments=delta.function_call.arguments,
                        created_at=func_ev.created_at,
                        source=source,
                        delta=True,
                    )
                )

            if chunk.choices[0].get("finish_reason") is not None:
                if assi_ev.content != "":
                    await self.__event(assi_ev)

                if func_ev is not None:
                    await self.__event(func_ev)

                if chunk.choices[0].finish_reason == "length":
                    await self.__event(
                        event.Error(
                            content="Max tokens exceeded",
                            source=assi_ev.id if func_ev is None else func_ev.id,
                        )
                    )

                if func_ev is not None and chunk.choices[0].finish_reason in [
                    "function_call",
                    "stop",
                ]:
                    result = await self.call_function(func_ev)
                    await self.__event(result)
                    await self.__invoke(result.id)

    async def call_function(self, source: event.FunctionCall) -> event.Event:
        """Call a function and put the result to the history."""

        try:
            args = json.loads(source.arguments)
        except Exception as err:
            return event.Error(
                content=(
                    f"Failed to parse arguments to call function `{source.name}`.\n>"
                    f" {err}\n\nGiven"
                    f" arguments:\n```json\n{source.arguments}\n```\n\nPlease fix the"
                    f" syntax and call `{source.name}` again."
                ),
                source=source.id,
            )

        match source.name:
            case "save_notes":
                return await self.call_save_notes(source, args)
            case "search_notes":
                return await self.call_search_notes(source, args)
            case "delete_notes":
                return await self.call_delete_notes(source, args)
            case "run_code":
                return await self.call_run_code(source, args)
            # case "generate_image":
            case _:
                return event.Error(
                    content=(
                        f"Unknown function: `{source.name}`\nPlease use only given"
                        " functions."
                    ),
                    source=source.id,
                )

    async def call_save_notes(
        self, source: event.FunctionCall, args: dict
    ) -> event.Event:
        if not isinstance(args.get("notes"), list) or any(
            [not isinstance(note, dict) for note in args["notes"]]
        ):
            return event.Error(
                content=(
                    "save_notes: `notes` argument must be a non-empty list of objects."
                ),
                source=source.id,
            )

        if any([not isinstance(note.get("content"), str) for note in args["notes"]]):
            return event.Error(
                content=(
                    "save_notes: `content` property of `notes` argument must be a"
                    " non-empty string."
                ),
                source=source.id,
            )

        if any(
            [
                note.get("available_term", "forever") not in TERMS
                for note in args["notes"]
            ]
        ):
            return event.Error(
                content=(
                    "save_notes: `available_term` property of `notes` argument must be"
                    " one of `a day`, `a month`, `a year`, or `forever`."
                ),
                source=source.id,
            )

        now = datetime.now(timezone.utc)
        notes = [
            Note(
                content=note["content"].strip(),
                created_at=now,
                expires_at=now
                + timedelta(days=TERMS[note.get("available_term", "forever")]),
            )
            for note in args["notes"]
            if len(note["content"].strip()) > 0
        ]

        if len(notes) == 0:
            return event.Error(
                content=(
                    "save_notes: `notes` argument must be a non-empty list of objects."
                ),
                source=source.id,
            )

        try:
            self.notes.save(self.user.id, notes)
        except Exception as err:
            return event.Error(
                content=f"save_notes: Failed to save notes.\n> {err}",
                source=source.id,
            )
        else:
            return event.FunctionOutput(
                name="save_notes",
                content=json.dumps({"result": "Succeed.", "saved_notes": len(notes)}),
                source=source.id,
            )

    async def call_search_notes(
        self, source: event.FunctionCall, args: dict
    ) -> event.Event:
        if "query" not in args:
            return event.Error(
                content="search_notes: `query` argument is required.",
                source=source.id,
            )

        if not isinstance(args["query"], str):
            return event.Error(
                content="search_notes: `query` argument must be a non-empty string.",
                source=source.id,
            )

        query = args["query"].strip()

        if len(query) == 0:
            return event.Error(
                content="search_notes: `query` argument must be a non-empty string.",
                source=source.id,
            )

        try:
            all_result = self.notes.query(
                self.user.id, query, n_results=100, threshold=0.2
            )
        except Exception as err:
            return event.Error(
                content=f"search_notes: Failed to search notes.\n> {err}",
                source=source.id,
            )

        if len(all_result) == 0:
            return event.FunctionOutput(
                name="search_notes",
                content="\n".join(
                    [
                        "{\n",
                        '  "result": "Found 0 notes.",',
                        '  "notes": [],',
                        (
                            '  "rule": "Before report user that not found notes, try'
                            ' again with different query at least 3 times.",'
                        ),
                        "}",
                    ]
                ),
                source=source.id,
            )

        n_tokens = 0
        for i, note in enumerate(all_result):
            n_tokens += note.n_tokens
            if n_tokens > 2048:
                limited_result = all_result[:i]
                break

        records = [
            json.dumps(
                {
                    "id": str(note.id),
                    "created_at": note.created_at.astimezone(self.timezone).isoformat(),
                    "content": note.content,
                }
            )
            for note in limited_result
        ]

        content = "\n".join(
            [
                "{\n",
                f'  "result": "Found {len(all_result)} notes.",',
                '  "notes": [',
                *[f"  {record}," for record in records],
                "  ],",
                (
                    '  "rule": "If there is no suitable notes found, please try'
                    ' different query before report to user.",'
                ),
                "}",
            ]
        )

        return event.FunctionOutput(
            name="search_notes",
            content=content,
            source=source.id,
        )

    async def call_delete_notes(
        self, source: event.FunctionCall, args: dict
    ) -> event.Event:
        if "ids" not in args:
            return event.Error(
                content="delete_notes: `ids` argument is required.",
                source=source.id,
            )

        if not isinstance(args["ids"], list):
            return event.Error(
                content="delete_notes: `ids` argument must be a non-empty list.",
                source=source.id,
            )

        try:
            ids = [uuid.UUID(id) for id in args["ids"]]
        except Exception as err:
            return event.Error(
                content=f"delete_notes: invalid `ids` argument: {err}",
                source=source.id,
            )

        try:
            self.notes.delete(self.user.id, ids)
        except Exception as err:
            return event.Error(
                content=f"delete_notes: Failed to delete notes.\n> {err}",
                source=source.id,
            )
        else:
            return event.FunctionOutput(
                name="delete_notes",
                content=json.dumps({"result": "Succeed.", "deleted_notes": len(ids)}),
                source=source.id,
            )

    async def call_run_code(
        self, source: event.FunctionCall, args: dict
    ) -> event.Event:
        supported_languages = ["python", "bash"]
        if "language" not in args or args.get("language") not in supported_languages:
            return event.Error(
                content=(
                    "run_code: `language` argument must be one of:"
                    f" {supported_languages}."
                ),
                source=source.id,
            )

        if (
            "code" not in args
            or not isinstance(args["code"], str)
            or len(args["code"].strip()) == 0
        ):
            return event.Error(
                content="run_code: `code` argument must be a non-empty string.",
                source=source.id,
            )

        language = args["language"]
        code = args["code"].strip()

        if language not in self.runners:
            self.runners[language] = CodeRunner(self.user.id, language)

        runner = self.runners[language]

        prev: event.Event | None = None
        async for ev in runner.execute(source.id, code):
            if prev is not None:
                await self.__event(prev)
            prev = ev

        if prev is None:
            return event.Error(
                content=(
                    "run_code: Failed to start Jupyter environment to execute code."
                ),
                source=source.id,
            )

        return prev


class EventReader:
    def __init__(self, thread: Thread):
        self.queue: asyncio.Queue[event.Event] = asyncio.Queue()
        self.thread = thread

    async def put(self, ev: event.Event) -> None:
        self.queue.put_nowait(ev)

    async def get(self) -> event.Event:
        return await self.queue.get()

    def __aiter__(self) -> AsyncIterator[event.Event]:
        return self

    async def __anext__(self) -> event.Event:
        return await self.get()

    def __enter__(self) -> Self:
        self.thread.subscribe(self.put)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.thread.unsubscribe(self.put)
