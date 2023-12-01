from langchain.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
import json
import psycopg2
from langchain.schema import AgentAction, AgentFinish, LLMResult
from typing import Any, Dict, List
from dotenv import load_dotenv
import os

load_dotenv()

class WSHandler(AsyncCallbackHandler):
    def __init__(self, websocket, session_id: str, user_q: str):
        self.session_id = session_id
        self.user_q = user_q
        self.websocket = websocket
        super().__init__()

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        obj = {
            "tool": action.tool,
            "tool_input": json.dumps(action.tool_input),
            "log": action.log
        }
        actionJson = json.dumps(obj)
        log(self.session_id, self.user_q, "on_agent_action", actionJson)
        await self.websocket.send_json({
            "callback": "on_agent_action",
            "thought": actionJson
        })

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        log(self.session_id, self.user_q, "on_agent_finish", json.dumps(finish))
        finishJson = json.loads(json.dumps(finish))
        await self.websocket.send_json({
            "callback": "on_agent_finish",
            "thought": finishJson
        })

class CustomHandler(BaseCallbackHandler):
    def __init__(self, session_id: str, user_q: str):
        self.session_id = session_id
        self.user_q = user_q
        super().__init__()

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        log(self.session_id, self.user_q, "on_agent_action", json.dumps(action))
        print(f"Callback: on_agent_action: {json.dumps(action, indent=2)}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        print("on_agent_finish")
        log(self.session_id, self.user_q, "on_agent_finish", json.dumps(finish))
        print(f"Callback: on_agent_finish: {json.dumps(finish, indent=2)}")

def log(session_id: str, user_q: str, callback_type: str, log_json: str) -> None:
    postgresUser = str(os.getenv("POSTGRES_USER"))
    postgresPassword = str(os.getenv("POSTGRES_PASSWORD"))
    postgresHost = str(os.getenv("POSTGRES_HOST"))
    postgresPort = str(os.getenv("POSTGRES_PORT"))
    conn = psycopg2.connect(
        host=postgresHost,
        port=postgresPort,
        database="chat_history",
        user=postgresUser,
        password=postgresPassword
    )
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO agent_log (session_id, user_q, callback_type, log) VALUES (%s, %s, %s, %s)",
        (session_id, user_q, callback_type, log_json)
    )
    conn.commit()
    cur.close()
    conn.close()