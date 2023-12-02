import logging
import os

import fire

from ..evaluation import evaluate_model
from ..inference import get_model_class
from .metrics import summarization_metrics

from .huggingface import (
    Text2TextSummarizer,
    CausalLMSummarizer,
    InstructText2TextSummarizer,
    InstructCausalLMSummarizer,
    AlpacaSummarizer,
    VicunaSummarizer,
    Llama2Summarizer,
    LlamaChatSummarizer,
    FalconChatSummarizer,
)
from .openai import OpenAISummarizer
from .cohere import CohereSummarizer

logger = logging.getLogger(__name__)


MODEL_MAP = {
    "gpt-[-\d\w]*": OpenAISummarizer,
    "facebook/opt-[\d\w]+": CausalLMSummarizer,
    ".*llama-?2.*chat.*": LlamaChatSummarizer,
    ".*llama-?2.*": Llama2Summarizer,
    ".*llama.*": CausalLMSummarizer,
    "bigscience/T0[_\d\w]*": InstructText2TextSummarizer,
    "google/flan-t5[-\d\w]+": InstructText2TextSummarizer,
    "google/long-t5[-\d\w]+": InstructText2TextSummarizer,
    ".*alpaca.*": AlpacaSummarizer,
    ".*vicuna.*": VicunaSummarizer,
    "summarize-((medium)|(xlarge))": CohereSummarizer,
    "mosaicml/mpt[-\d\w]$": CausalLMSummarizer,
    "tiiuae/falcon[-\d\w]+chat": FalconChatSummarizer,
    "tiiuae/falcon[-\d\w]+instruct": InstructCausalLMSummarizer,
    "tiiuae/falcon[-\d\w]$": CausalLMSummarizer,
    "mosaicml/mpt[-\d\w]+instruct": AlpacaSummarizer,
}


def get_summarizer_model_class(
    model_name, model_map=MODEL_MAP, default_class=Text2TextSummarizer
):
    model_class = get_model_class(
        model_name, model_map=model_map, default_class=default_class
    )
    return model_class


def evaluate_summarizer(model_name=None, metrics=None, **kwargs):
    model_class = kwargs.pop("model_class", None)
    if model_name and model_class is None:
        model_class = get_summarizer_model_class(model_name)
    if metrics is None:
        metrics = []
    metrics.append(summarization_metrics)
    result = evaluate_model(
        model_name=model_name,
        model_class=model_class,
        metrics=metrics,
        **kwargs,
    )
    return result


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire(evaluate_summarizer)
