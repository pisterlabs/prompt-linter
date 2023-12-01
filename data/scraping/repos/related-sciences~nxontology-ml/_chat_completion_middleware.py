import logging
import os
from collections import Counter
from collections.abc import Callable, Iterable
from copy import copy
from os import _Environ
from pathlib import Path
from string import Formatter

import dotenv
import openai

from nxontology_ml.gpt_tagger._models import TaskConfig
from nxontology_ml.gpt_tagger._openai_models import (
    OPENAI_API_KEY,
    OPENAI_MODELS,
    ChatCompletionMessage,
    ChatCompletionsPayload,
    Response,
)
from nxontology_ml.gpt_tagger._utils import (
    counter_or_empty,
    log_json_if_enabled,
)
from nxontology_ml.utils import ROOT_DIR

CREATE_FN_TYPE = Callable[[ChatCompletionsPayload], Response]


class _ChatCompletionMiddleware:
    """
    Thin wrapper around OpenAi's ChatCompletion API.

    Allows to:
    - Handle prompt templating
    - Make testing easier
    - Instrument API usage

    Resources:
    - Chat GPT use: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    """

    def __init__(
        self,
        partial_payload: ChatCompletionsPayload,
        prompt_template: str,
        create_fn: CREATE_FN_TYPE,
        logs_path: Path | None,
        counter: Counter[str],
    ):
        """
        Intended to get constructed using cls.from_config(config)
        """
        self._verify_partial_payload(partial_payload)
        self._verify_prompt_template(prompt_template)
        self._partial_payload = partial_payload
        self._prompt_template = prompt_template
        self._create_fn = create_fn
        self._logs_path = logs_path
        self._counter = counter

    def create(self, records: Iterable[str]) -> Response:
        record_list: list[str] = list(records)
        payload: ChatCompletionsPayload = copy(self._partial_payload)
        payload["messages"][-1]["content"] = self._prompt_template.format(
            records="\n".join(record_list)
        ).strip()
        self._counter["ChatCompletion/create_requests"] += 1
        self._counter["ChatCompletion/records_processed"] += len(record_list)
        logging.debug(f"Sending {len(record_list)} to OpenAI's ChatCompletion API")
        log_json_if_enabled(self._logs_path, "requests", payload)
        resp = self._create_fn(**payload)  # type: ignore
        log_json_if_enabled(self._logs_path, "responses", resp)
        # FIXME: Would we want to support async io?
        return resp

    @classmethod
    def from_config(
        cls,
        config: TaskConfig,
        counter: Counter[str] | None = None,
        env: _Environ[str] = os.environ,
    ) -> "_ChatCompletionMiddleware":
        """
        Builder class from config
        Env is exposed because OpenAI's SDK implicitly depends on the API key
        """
        openai.api_key = env.get(OPENAI_API_KEY, None) or dotenv.get_key(
            dotenv_path=ROOT_DIR / ".env",
            key_to_get=OPENAI_API_KEY,
        )
        partial_payload = ChatCompletionsPayload(
            model=config.openai_model_name,
            messages=[ChatCompletionMessage(role="user", content="__PLACEHOLDER__")],
        )
        if config.model_temperature:
            partial_payload["temperature"] = config.model_temperature
        if config.model_top_p:
            partial_payload["top_p"] = config.model_top_p
        if config.model_n:
            partial_payload["n"] = config.model_n
        # At the moment, only chat_completion is supported.
        #  See: https://openai.com/blog/gpt-4-api-general-availability
        return cls(
            partial_payload=partial_payload,
            prompt_template=config.prompt_path.read_text(),
            create_fn=openai.ChatCompletion.create,
            logs_path=config.logs_path,
            counter=counter_or_empty(counter),
        )

    @staticmethod
    def _verify_prompt_template(prompt_template: str) -> None:
        fields = {t[1] for t in Formatter().parse(prompt_template) if t[1]}
        if "records" not in fields:
            raise ValueError(
                'Invalid prompt provided: Template key "{records}" must be present.'
            )

    @staticmethod
    def _verify_partial_payload(partial_payload: ChatCompletionsPayload) -> None:
        model = partial_payload.get("model", "MISSING")
        if model not in OPENAI_MODELS:
            raise ValueError(f"Unsupported OpenAI Model: {model}")

        messages = partial_payload.get("messages", [])
        if not len(messages) > 0:
            raise ValueError("Invalid partial_payload: Should contain message(s)")
