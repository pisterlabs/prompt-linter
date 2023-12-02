from dataclasses import dataclass
from enum import Enum

from langchain.llms.base import LLM


class Language(Enum):
    English = '英语'
    Chinese = '中文'
    Korean = '韩语'
    Russian = '俄语'
    Cantonese = '粤语'
    French = '法语'
    Japanese = '日语'
    Spanish = '西班牙语'
    Portuguese = '葡萄牙语'

    @classmethod
    def list(cls):
        return [lang.value for lang in cls]


@dataclass
class ModelConf:
    llm_model: LLM
    temperature: float = None
    top_k: int = None
    top_p: float = None
