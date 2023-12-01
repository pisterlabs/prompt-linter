import requests 
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Sources:
# https://python.langchain.com/docs/modules/model_io/models/llms/custom_llm
# https://github.com/redhat-et/foundation-models-for-documentation/blob/master/notebooks/langchain-api-client.ipynb
# https://github.com/HunterGerlach/deep-thought/blob/main/src/hosted_llm.py

class CustomLLM(LLM):
    api_url: str
    "The URL of the API endpoint."

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        """Call the LLM."""
        response = requests.get(self.api_url,
                                params={"text" : prompt},
                                timeout=600)
        if response.status_code == 200:
            return str(response.content).split("[/INST]")[1]
        return f"Model Server is not Working due to error {response.status_code}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"api_url": self.api_url}