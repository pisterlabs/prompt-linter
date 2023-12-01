import tiktoken
from tiktoken import Encoding

from nxontology_ml.gpt_tagger._models import TaskConfig
from nxontology_ml.gpt_tagger._openai_models import OPENAI_MODELS


class _TiktokenBatcher:
    """
    Batch records based on Tiktoken token count: Will buffer records until the buffer is full.

    Uses:
    - https://github.com/openai/tiktoken
    - (Example of tiktoken: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)
    """

    def __init__(
        self,
        max_token_cnt: int,
        tiktoken_encoding: Encoding,
        record_buffer: list[str] | None = None,
        token_initial_cnt: int = 0,
    ):
        """
        Intended to get constructed using cls.from_config(config)

        Args:
            max_token_cnt: Size of the buffer (in terms of # of token)
            tiktoken_encoding: OpenAI Encoding to be used to count the tokens
            record_buffer:
            token_initial_cnt:
        """
        self._max_token_cnt = max_token_cnt
        self._tiktoken_encoding = tiktoken_encoding
        self._record_buffer = record_buffer or []
        self._buffer_token_cnt = token_initial_cnt

    def add_record_to_buffer(self, record: str) -> list[str] | None:
        """
        Add one record to the buffer
        Returns the content of the buffer if it is full (i.e. ready to process)
        """
        if self._buffer_is_full(next_record=record):
            assert len(self._record_buffer) > 0
            return self.flush_buffer(next_record=record)
        self._do_add_record_to_buffer(record)
        return None

    def flush_buffer(self, next_record: str | None = None) -> list[str]:
        """
        Flush the content of the buffer (even if not full).

        Args:
            next_record: First record of the next buffer (if provided)

        Returns:
            Content of the flushed buffer
        """
        old_buffer = self._record_buffer
        self._record_buffer = []
        self._buffer_token_cnt = 0
        if next_record:
            self._do_add_record_to_buffer(next_record)
        return old_buffer

    def _do_add_record_to_buffer(self, record: str) -> None:
        """
        Add record to buffer and update _buffer_token_cnt
        """
        self._record_buffer.append(record)
        self._buffer_token_cnt += self._get_token_cnt_from_record(record)
        if self._buffer_is_full():
            raise ValueError(
                f"Buffer size exceeded: {self._buffer_token_cnt=} > {self._max_token_cnt=}"
            )

    def _get_token_cnt_from_record(self, record: str) -> int:
        return len(self._tiktoken_encoding.encode(record))

    def _buffer_is_full(self, next_record: str | None = None) -> bool:
        """
        Returns True if:
        - It is already full
        - If it is not full but doesn't have capacity for the next_record
        """
        token_cnt: int = self._buffer_token_cnt
        if next_record:
            token_cnt += self._get_token_cnt_from_record(next_record)
        return token_cnt > self._max_token_cnt

    @classmethod
    def from_config(cls, config: TaskConfig) -> "_TiktokenBatcher":
        tiktoken_encoding = tiktoken.encoding_for_model(config.openai_model_name)
        prompt_token_cnt = len(tiktoken_encoding.encode(config.prompt_path.read_text()))
        assert 0.0 < config.prompt_token_ratio < 1.0, print(
            f"Wrong {config.prompt_token_ratio=} value."
        )
        max_token_cnt = (
            int(
                OPENAI_MODELS[config.openai_model_name].max_token_cnt
                * config.prompt_token_ratio
            )
            - prompt_token_cnt
        )
        if max_token_cnt <= 0:
            raise ValueError(
                "The provided prompt has more tokens than the window of the model."
            )
        return cls(
            max_token_cnt=max_token_cnt,
            tiktoken_encoding=tiktoken_encoding,
        )
