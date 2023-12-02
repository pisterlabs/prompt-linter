from langchain.llms import CTransformers

from dataclasses import dataclass


@dataclass
class GGMLConfig:
    model_path: str
    model_type: str
    max_new_tokens: int
    temperature: float = 0.7
    gpu_layers: int = 0


def build_ggml_llm(config: GGMLConfig) -> CTransformers:
    llm = CTransformers(
        model=config.model_path,
        model_type=config.model_type,
        config={
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "gpu_layers": config.gpu_layers,
        },
    )
    return llm
