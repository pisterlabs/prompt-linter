from pathlib import Path
from guidance.llms import Transformers
from huggingface_hub import snapshot_download

from transformers import LlamaTokenizer

from exllama_hf import ExllamaHF


class ExLLaMA(Transformers):
    """A HuggingFace transformers version of the LLaMA language
    model with Guidance support."""

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):
        # load the LLaMA specific tokenizer and model

        assert tokenizer is None, "We will not respect any tokenizer from the caller."
        assert isinstance(model, str), "Model should be a str with LLaMAGPTQ"

        print(f"Initializing ExLlamaGPTQ with model {model}")

        models_dir = "./models"
        name_suffix = model.split("/")[1]
        model_dir = f"{models_dir}/{name_suffix}"
        snapshot_download(repo_id=model, local_dir=model_dir)

        (model, tokenizer) = _load(model_dir)

        print(f"Loading tokenizer from: {model_dir}")

        tokenizer = LlamaTokenizer.from_pretrained(Path(model_dir))

        return super()._model_and_tokenizer(model, tokenizer, **kwargs)

    @staticmethod
    def role_start(role):
        if role == "user":
            return "USER: "
        elif role == "assistant":
            return "ASSISTANT: "
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return ""
        elif role == "assistant":
            return "</s>"
        else:
            return ""


def _load(model_dir: str):
    model_dir = Path(model_dir)

    exllama_hf = ExllamaHF.from_pretrained(model_dir)

    return (exllama_hf, None)


# Config found from gptq:
# config: LlamaConfig {
#   "_name_or_path": "models/tulu-13B-GPTQ",
#   "architectures": [
#     "LlamaForCausalLM"
#   ],
#   "bos_token_id": 1,
#   "eos_token_id": 2,
#   "hidden_act": "silu",
#   "hidden_size": 5120,
#   "initializer_range": 0.02,
#   "intermediate_size": 13824,
#   "max_position_embeddings": 2048,
#   "model_type": "llama",
#   "num_attention_heads": 40,
#   "num_hidden_layers": 40,
#   "pad_token_id": 0,
#   "rms_norm_eps": 1e-06,
#   "tie_word_embeddings": false,
#   "torch_dtype": "float32",
#   "transformers_version": "4.30.2",
#   "use_cache": true,
#   "vocab_size": 32001
# }

# config = {
#   "_name_or_path": "models/tulu-13B-GPTQ",
#   "architectures": [
#     "LlamaForCausalLM"
#   ],
#   "bos_token_id": 1,
#   "eos_token_id": 2,
#   "hidden_act": "silu",
#   "hidden_size": 5120,
#   "initializer_range": 0.02,
#   "intermediate_size": 13824,
#   "max_position_embeddings": 2048,
#   "model_type": "llama",
#   "num_attention_heads": 40,
#   "num_hidden_layers": 40,
#   "pad_token_id": 0,
#   "rms_norm_eps": 1e-06,
#   "tie_word_embeddings": False,
#   "torch_dtype": "float32",
#   "transformers_version": "4.30.2",
#   "use_cache": True,
#   "vocab_size": 32001
# }
