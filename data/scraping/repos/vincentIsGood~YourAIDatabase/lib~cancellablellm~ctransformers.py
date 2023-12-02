from typing import Any, List, Sequence

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms import CTransformers


class CancellableLLM(CTransformers):
    stopRequested = False

    def stopGen(self):
        self.stopRequested = True
    
    def _call(
            self, prompt: str, 
            stop: 'Sequence[str] | None' = None, 
            run_manager: 'CallbackManagerForLLMRun | None' = None, 
            **kwargs: Any) -> str:
        # Modified implementation of CTransformers._call
        self.stopRequested = False
        text = []
        _run_manager = run_manager or CallbackManagerForLLMRun.get_noop_manager()
        for chunk in self.client(prompt, stop=stop, stream=True):
            if self.stopRequested:
                return "".join(text)
            text.append(chunk)
            _run_manager.on_llm_new_token(chunk, verbose=self.verbose)
        return "".join(text)