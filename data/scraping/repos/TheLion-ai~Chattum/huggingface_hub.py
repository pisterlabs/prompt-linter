import requests
from langchain.llms import HuggingFaceHub

from .base_model import LLMTemplate, UserVariable


def load_available_models() -> list[str]:
    models = requests.get(
        "https://api-inference.huggingface.co/framework/text-generation-inference"
    ).json()
    return [model["model_id"] for model in models]


class HugggingfaceHubModel(LLMTemplate):
    """LLM created by OpenAI."""

    name: str = "Huggingface Hub"
    user_description = "Model from Huggingface Hub"
    model_type: str = "llm"
    supports_tools: bool = False

    user_variables: list[UserVariable] = [
        UserVariable(
            name="repo_id",
            description="name of the model's repository on huggingface",
            form_type="text",
            default_value=None,
            available_values=load_available_models(),
        ),
        UserVariable(
            name="huggingfacehub_api_token",
            description="copy the api token form https://huggingface.co/settings/tokens",
            form_type="text",
        ),
        # UserVariable(name="stream", description="???", form_type="text"),
        # UserVariable(name="n", description="???", form_type="text"),
        UserVariable(
            name="temperature",
            description="temperature for the model",
            form_type="float",
            default_value=0.9,
        ),
    ]

    def as_llm(self) -> HuggingFaceHub:
        """Return the LLM."""
        return HuggingFaceHub(
            repo_id=self.variables_dict["repo_id"],
            huggingfacehub_api_token=self.variables_dict["huggingfacehub_api_token"],
            model_kwargs={"temperature": self.variables_dict["temperature"]},
        )
