# Import base deps
import reflex as rx
from pydantic import BaseModel

# Import IBM Gen
from genai.model import Credentials

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from genai.extensions.langchain import LangChainInterface


# Get reflex config
config = rx.config.get_config()


class WatsonxWorkbenchLangchainLLM(BaseModel):
    generate_params: dict = {GenParams.MAX_NEW_TOKENS: 25}

    model_id: str = "google/flan-ul2"

    def from_pretrained(self):
        creds = Credentials(
            config.watsonx_workbench_api_key,
            api_endpoint=config.watsonx_workbench_api_endpoint,
        )

        llm = LangChainInterface(
            model=self.model_id,
            credentials=creds,
            params=self.generate_params,
        )

        return llm
