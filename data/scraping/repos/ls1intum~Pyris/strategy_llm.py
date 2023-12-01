import logging
from typing import Dict, Any

from guidance.llms import LLM, LLMSession

from app.config import settings, OpenAIConfig
from app.services.cache import cache_store
from app.services.circuit_breaker import CircuitBreaker

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class StrategyLLM(LLM):
    llm_keys: list[str]
    llm_configs: dict[str, OpenAIConfig]
    llm_sessions: dict[str, LLMSession]

    def __init__(self, llm_keys: list[str]):
        super().__init__()
        if llm_keys.__len__() == 0:
            raise ValueError("No LLMs configured")
        self.llm_keys = [
            llm_keys
            for llm_keys in llm_keys
            if llm_keys in settings.pyris.llms
            and isinstance(settings.pyris.llms[llm_keys], OpenAIConfig)
        ]
        self.llm_configs = {
            llm_key: settings.pyris.llms[llm_key] for llm_key in self.llm_keys
        }
        self.llm_instances = {
            llm_key: llm_config.get_instance()
            for llm_key, llm_config in self.llm_configs.items()
        }
        self.llm_sessions = {}

    def first_llm(self):
        return self.llm_instances[self.llm_keys[0]]

    def extract_function_call(self, text):
        return self.first_llm().extract_function_call(text)

    def __call__(self, *args, asynchronous=False, **kwargs):
        return self.first_llm()(*args, asynchronous=asynchronous, **kwargs)

    def __getitem__(self, key):
        return self.first_llm()[key]

    def session(self, asynchronous=False):
        return StrategyLLMSession(self)

    def encode(self, string, **kwargs):
        return self.first_llm().encode(string)

    def decode(self, tokens, **kwargs):
        return self.first_llm().decode(tokens)

    def id_to_token(self, id):
        return self.first_llm().id_to_token(id)

    def token_to_id(self, token):
        return self.first_llm().token_to_id(token)

    def role_start(self, role_name, **kwargs):
        return self.first_llm().role_start(role_name, **kwargs)

    def role_end(self, role=None):
        return self.first_llm().role_end(role)

    def end_of_text(self):
        return self.first_llm().end_of_text()


class StrategyLLMSession(LLMSession):
    llm: StrategyLLM
    current_session_key: str
    current_session: LLMSession

    def __init__(self, llm: StrategyLLM):
        super().__init__(llm)
        self.llm = llm

    def get_total_context_length(
        self, prompt: str, llm_key: str, max_tokens: int
    ):
        prompt_token_length = (
            self.llm.llm_instances[llm_key].encode(prompt).__len__()
        )
        return prompt_token_length + max_tokens

    def set_correct_session(
        self, prompt: str, max_tokens: int, exclude_llms=None
    ):
        if exclude_llms is None:
            exclude_llms = []

        selected_llm = None

        for llm_key in self.llm.llm_keys:
            if (
                llm_key not in exclude_llms
                and cache_store.get(llm_key + ":status") != "OPEN"
                and self.llm.llm_configs[llm_key].spec.context_length
                >= self.get_total_context_length(prompt, llm_key, max_tokens)
            ):
                selected_llm = llm_key
                break

        if selected_llm is None:
            log.error(
                "No viable LLMs found! Using LLM with longest context length."
            )
            llm_configs = {
                llm_key: config
                for llm_key, config in self.llm.llm_configs.items()
                if llm_key not in exclude_llms
            }
            if llm_configs.__len__() == 0:
                raise ValueError("All LLMs are excluded!")
            selected_llm = max(
                llm_configs,
                key=lambda llm_key: llm_configs[llm_key].spec.context_length,
            )

        log.info("Selected LLM: " + selected_llm)

        self.current_session_key = selected_llm
        if selected_llm in self.llm.llm_sessions:
            self.current_session = self.llm.llm_sessions[selected_llm]
        else:
            self.current_session = self.llm.llm_instances[
                selected_llm
            ].session(asynchronous=True)
            self.llm.llm_sessions[selected_llm] = self.current_session

    async def __call__(self, *args, **kwargs):
        prompt = args[0]
        max_tokens = kwargs["max_tokens"]
        log.info(
            f"Trying to make request using strategy "
            f"llm with llms [{self.llm.llm_keys}]"
        )
        self.set_correct_session(prompt, max_tokens)

        def call():
            return self.current_session(*args, **kwargs)

        exclude_llms = []

        while True:
            try:
                response = await CircuitBreaker.protected_call_async(
                    func=call, cache_key=self.current_session_key
                )
                return response
            except Exception as e:
                log.error(
                    f"Exception '{e}' while making request! "
                    f"Trying again with a different LLM."
                )
                exclude_llms.append(self.current_session_key)
                self.set_correct_session(prompt, max_tokens, exclude_llms)

    def __exit__(self, exc_type, exc_value, traceback):
        return self.current_session.__exit__(exc_type, exc_value, traceback)

    def _gen_key(self, args_dict):
        return self.current_session._gen_key(args_dict)

    def _cache_params(self, args_dict) -> Dict[str, Any]:
        return self.current_session._cache_params(args_dict)
