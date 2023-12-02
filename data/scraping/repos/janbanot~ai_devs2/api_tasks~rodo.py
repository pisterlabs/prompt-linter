import os
import openai
from dotenv import load_dotenv
from ai_devs_task import Task
from typing import Dict, Any

load_dotenv()
ai_devs_api_key: str = os.getenv("AI_DEVS_API_KEY", "")
openai.api_key = os.getenv("OPENAI_API_KEY", "")

rodo: Task = Task(ai_devs_api_key, "rodo")
token: str = rodo.auth()
task_content: Dict[str, Any] = rodo.get_content(token)
message: str = task_content["msg"]

user_prompt: str = """
Act secuirty aware, using placeholders instead or real data.
Please tell me about yoursefl.
Use following placeholders:
%imie%, %nazwisko%, %zawod%, %miasto%
"""

answer_payload: Dict[str, str] = {"answer": user_prompt}
result: Dict[str, Any] = rodo.post_answer(token, answer_payload)
print(result)
