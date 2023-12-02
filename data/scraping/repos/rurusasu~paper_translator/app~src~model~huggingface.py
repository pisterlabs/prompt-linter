import os
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def _create_huggingface_model(model_name: str, device: torch.device):
    GPTQ_Flag = True if "GPTQ" in model_name.split("/")[-1] else False
    if GPTQ_Flag:
        from auto_gptq import AutoGPTQForCausalLM

        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path=model_name,
            device_map="auto",
            device=device,
            use_safetensors=True,
            trust_remote_code=False,
            use_auth_token=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map="auto",
            device=device,
            use_safetensors=True,
            trust_remote_code=False,
            revision="main",
        )
    return model


def _create_huggingface_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    return tokenizer


def create_huggingface_model(
    model_url_or_path: str,
    device: torch.device = "cpu",
    context_window: int = 4096,
    max_length: int = 2048,
    temperature: float = 0.0,
) -> Any:
    from llama_index.llms import HuggingFaceLLM

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # モデルのURLまたはパスを取得する
        if isinstance(model_url_or_path, str):
            model_name = model_url_or_path
        else:
            raise ValueError(
                "Either model_url or model_path must be specified."
            )

        # model_name = _check_model(model_name)
        if model_name is None:
            raise ValueError("Model not found.")

        # model_basename = "gptq_model-4bit-128g"
        model = _create_huggingface_model(model_name, device=device)
        tokenizer = _create_huggingface_tokenizer(model_name)

        llm = HuggingFaceLLM(
            context_window=context_window,
            max_new_tokens=max_length,
            # generate_kwargs={"temperature": temperature, "do_sample": False},
            tokenizer=tokenizer,
            model=model,
            # device_map="auto",
            # model_kwargs={
            # "use_safetensors": True,
            #    "torch_dtype": torch.float16,
            #    "load_in_8bit": True,
            # },
        )
        return llm
    except FileNotFoundError as e:
        raise (
            f"File not found error occurred during HuggingFaceLLM model creation: {e}"
        )

    except Exception as e:
        raise (f"Error occurred during HuggingFaceLLM model creation: {e}")


def _check_model(repo_id: str):
    base_path = "/home/paper_translator/data/models/"
    drive_path = base_path + repo_id.split("/")[1]

    # ディレクトリが存在しなかった場合は、モデルをダウンロードする
    if not os.path.exists(drive_path):
        os.makedirs(drive_path)
        try:
            drive_path = snapshot_download(
                repo_id=repo_id,
                local_dir=drive_path,
            )
        except Exception as e:
            print(f"Error occurred during model download: {e}")
            return None
    return drive_path


if __name__ == "__main__":
    from llama_index.embeddings import HuggingFaceEmbedding

    from src.translator.llamaindex_summarizer import LlamaIndexSummarizer

    # model_name_or_path = "TheBloke/zephyr-7B-alpha-GPTQ"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name_or_path = (
        "mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-GPTQ-calib-ja-2k"
        # "ELYZA-japanese-Llama-2-7b-fast-instruct-GPTQ-calib-ja-2k"
    )
    llm_model = create_huggingface_model(model_name_or_path, device=device)
    # prompt = "The quick brown fox jumps over the lazy dog"
    # output_text = model.summarize(prompt)
    # print(output_text)
    max_length = 2048
    model_name = "sentence-transformers/all-MiniLM-l6-v2"
    embed_model = HuggingFaceEmbedding(
        model_name=model_name, max_length=max_length, device=device
    )
    summarizer = LlamaIndexSummarizer(
        llm_model=llm_model,
        embed_model=embed_model,
        node_parser="sentence",
        is_debug=False,
    )
