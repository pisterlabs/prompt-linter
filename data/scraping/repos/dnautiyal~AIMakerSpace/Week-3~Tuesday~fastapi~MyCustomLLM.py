from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from run import generate_text, get_task_status
import time

class FastApiLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "FastApiLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        if len(prompt) <= 250:
          return prompt
        try:
          task_id = generate_text(prompt)
          while True:
            status = get_task_status(task_id)
            if "Task Pending" not in status:
              print(f"Task-id = {task_id} is ready. Time taken = {status['time']}, Memory used = {status['memory']}")
              return status["result"]
            time.sleep(2)
        except:
          return "FastApiLLM: Exception encountered"

    