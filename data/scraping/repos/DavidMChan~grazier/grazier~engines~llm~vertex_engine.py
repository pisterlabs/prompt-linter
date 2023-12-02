try:
    from vertexai.preview.language_models import TextGenerationModel
except ImportError:
    TextGenerationModel = None

from typing import Any, Dict, List

from ratelimit import limits, sleep_and_retry

from grazier.engines.llm import LLMEngine, register_engine
from grazier.utils.python import singleton


class VertexLLMEngine(LLMEngine):
    def __init__(self, model: str, **kwargs: Dict[str, Any]) -> None:
        if TextGenerationModel is None:
            raise ImportError("Please install the Vertex AI SDK to use this LM engine.")

        self._model = TextGenerationModel.from_pretrained(model)
        self._parameters = {
            # Token limit determines the maximum amount of text output.
            "max_output_tokens": kwargs.pop("max_output_tokens", kwargs.pop("max_tokens", 256)),
            "top_p": 0.95,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        } | kwargs

    @sleep_and_retry  # type: ignore
    @limits(  # type: ignore
        calls=40, period=60
    )  # This is the default rate limit for Vertex AI (actual rate limit is 60 calls per minute, but we'll be conservative)
    def _rate_limited_model_predict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.predict(*args, **kwargs)

    def call(self, prompt: str, n_completions: int = 1, **kwargs: Any) -> List[str]:
        # Normalize kwargs from openai to vertexai (some common parameters are different)
        kwargs = (
            self._parameters
            | {
                "max_output_tokens": kwargs.pop("max_output_tokens", kwargs.pop("max_tokens", 256)),
                "temperature": kwargs.pop("temperature", 1.0),
            }
            | kwargs
        )

        return [self._rate_limited_model_predict(prompt, **kwargs).text for _ in range(n_completions)]  # type: ignore

    @staticmethod
    def is_configured() -> bool:
        # Check to see if the Vertex AI SDK is installed, and if so, if the user has configured their credentials.
        if TextGenerationModel is None:
            return False

        # Check to see if the user has configured their google cloud credentials.
        try:
            from google.auth import default
        except ImportError:
            return False

        try:
            default()
        except Exception:
            return False

        return True


@register_engine
@singleton
class PaLMEngine(VertexLLMEngine):
    name = ("PaLM", "palm")

    def __init__(self) -> None:
        super().__init__("text-bison@001")
