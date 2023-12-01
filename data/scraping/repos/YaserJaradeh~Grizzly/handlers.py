import asyncio
from typing import Any, Optional
from uuid import UUID

from fastapi import WebSocket
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish

from app.services.ws.manager import ConnectionManager


def _sanitize_text(text: str) -> str:
    # Remove ===> from the text
    text = text.replace("===> ", "")
    # Replace dataframe with comparison
    text = text.replace("dataframe", "comparison")
    text = text.replace("DataFrame", "comparison")
    # Replace column with property
    text = text.replace("columns", "properties")
    text = text.replace("Columns", "properties")
    text = text.replace("column", "property")
    text = text.replace("Column", "property")
    # Replace "Agent stopped due to iteration limit or time limit" with "Sorry! I wasn't able to find an answer."
    text = text.replace(
        "Agent stopped due to iteration limit or time limit",
        "Sorry! I wasn't able to find an answer.",
    )
    # Replace "I do not know" with "Uh-oh! I don't know the answer to that."
    text = text.replace(
        "Sorry!, I do not know", "Uh-oh! I don't know the answer to that."
    )
    #################################
    # Replace Action: json_spec_list_keys' with "I should check the properties of the comparison first."
    text = text.replace(
        "Action: json_spec_list_keys",
        "I should check the properties of the comparison first.",
    )
    return text


class AsyncStreamThoughtsAndAnswerHandler(AsyncCallbackHandler):
    def __init__(self):
        self._action_logs = asyncio.Queue()
        self.done = asyncio.Event()

    # This is an async generator
    async def action_logs(self):
        while True:
            if self.done.is_set():  # check if the done event is set
                break
            log = (
                await self._action_logs.get()
            )  # this will block until there is something in the queue
            yield log
            self._action_logs.task_done()  # notify that the task is done

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        print(action)
        self._action_logs.put_nowait(
            _sanitize_text(
                {"thought": self._extract_thought_from_log(action.log)}.__str__() + "\n"
            )
        )  # this will unblock the action_logs generator

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._action_logs.put_nowait(
            _sanitize_text({"answer": finish.return_values["output"]}.__str__() + "\n")
        )  # this will unblock the action_logs generator
        self.done.set()  # set the done event

    @staticmethod
    def _extract_thought_from_log(log: str) -> str:
        """Extract the thought from the log."""
        log = log.strip()
        if len(thoughts := log.split("\n")) > 0 and thoughts[0].startswith("Thought:"):
            return thoughts[0].replace("Thought:", "").strip()
        else:
            return "===> " + thoughts[0]


class WSStreamThoughtsAndAnswerHandler(BaseCallbackHandler):
    def __init__(self, websocket: WebSocket, ws_manager: ConnectionManager):
        self.websocket = websocket
        self.ws_manager = ws_manager

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        await self.ws_manager.reply(
            self.websocket,
            _sanitize_text(
                {"thought": self._extract_thought_from_log(action.log)}.__str__() + "\n"
            ),
        )

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        await self.ws_manager.reply(
            self.websocket,
            _sanitize_text({"answer": finish.return_values["output"]}.__str__() + "\n"),
        )

    @staticmethod
    def _extract_thought_from_log(log: str) -> str:
        """Extract the thought from the log."""
        log = log.strip()
        if len(thoughts := log.split("\n")) > 0 and thoughts[0].startswith("Thought:"):
            return thoughts[0].replace("Thought:", "").strip()
        else:
            return "===> " + thoughts[0]
