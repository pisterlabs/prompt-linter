import structlog
from guidance.llms import LLM, Mock, OpenAI

from wet_toast_talk_radio.scriptwriter.config import LLMConfig

logger = structlog.get_logger()


def new_llm(cfg: LLMConfig) -> LLM:
    logger.info("Creating new LLM")

    if cfg.virtual:
        return Mock(output=cfg.fake_responses)
    else:
        return OpenAI(
            model=cfg.model,
            caching=False,
            api_key=cfg.openai_api_key.value(),
        )
