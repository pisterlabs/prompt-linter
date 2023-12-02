from abc import ABC
from langchain.llms.base import LLM
from typing import Optional, List
from src.models.loader import LoaderCheckPoint
from src.models.base import (BaseAnswer, AnswerResult)
from transformers.generation import GenerationConfig


class Qwen(BaseAnswer, LLM, ABC):
    checkPoint: LoaderCheckPoint = None
    history_len: int = 5
    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint
        self.checkPoint.model.generation_config = GenerationConfig(
            max_new_tokens=4096,
            temperature=0.3,
            top_k=0,
            top_p=0.8,
            repetition_penalty=1.1,
            do_sample=True,
            chat_format="chatml",
            eos_token_id=151643,
            pad_token_id=151643,
            max_window_size=6144,
        )

    @property
    def _llm_type(self) -> str:
        return "Qwen"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"__call:{prompt}")
        resp, _ = self.checkPoint.model.chat(
            self.checkPoint.tokenizer,
            prompt,
            history=[],
        )
        print(f"response:{resp}")
        print(f"+++++++++++++++++++++++++++++++++++")
        return resp

    def generatorAnswer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False):

        if streaming:
            history += [[]]
            position = 0
            for resp in self.checkPoint.model.chat_stream(
                    self.checkPoint.tokenizer,
                    prompt,
                    history=history[-self.history_len:] if self.history_len > 1 else [],
            ):
                print(resp[position:], end="", flush=True)
                history[-1] = [prompt, resp]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": resp}
                yield answer_result
            self.checkPoint.clear_torch_cache()
        else:
            resp, _ = self.checkPoint.model.chat(
                self.checkPoint.tokenizer,
                prompt,
                history=history[-self.history_len:] if self.history_len > 0 else [],
            )
            print(f"resp: {resp}")
            self.checkPoint.clear_torch_cache()
            history += [[prompt, resp]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": resp}
            yield answer_result


