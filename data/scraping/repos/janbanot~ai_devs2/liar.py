import os
import openai
from dotenv import load_dotenv
from ai_devs_task import Task
from typing import Dict, Any

load_dotenv()
ai_devs_api_key: str = os.getenv("AI_DEVS_API_KEY", "")
openai.api_key = os.getenv("OPENAI_API_KEY", "")

liar: Task = Task(ai_devs_api_key, "liar")

token: str = liar.auth()
task_content: Dict[str, Any] = liar.get_content(token)

question: Dict[str, str] = {"question": "What is the capital of Poland?"}
answer_json: Dict[str, Any] = liar.post_question(token, question)
answer: str = answer_json["answer"]

prompt: str = """
Answer simply YES or NO
Is it a correct answer to the following question:
"What is the capital of Poland?"
"""


check_answer: openai.ChatCompletion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": answer}
        ]
    )
answer_content: str = check_answer["choices"][0]["message"]["content"]

result_payload: Dict[str, str] = {"answer": answer_content}
result: Dict[str, Any] = liar.post_answer(token, result_payload)
print(result)
