"""
GPT: Generative Pre-Trained Transformer
- generates something (usually text)
- parameters already trained on books, Internet, etc.
- based on the transformer architecture

We'll implement GPT-2 and grab pre-trained parameters from OpenAI so that we can
input text and generate new text.

Based on [GPT in 60 Lines of NumPy | Jay Mody](https://jaykmody.com/blog/gpt-from-scratch/).

mamba install regex requests tqdm typed-argument-parser tensorflow==2.11.0

# Optional
mamba install pretty_errors black
python -m pretty_errors
"""

# TODO:
# - play around with input files (maybe plot wpe?)

import json
import re
from pathlib import Path

import numpy as np
import requests
from encoder import Encoder, Tokens, get_encoder
from tap import tapify
from tensorflow import train as tf_train
from tqdm import tqdm


def __download_gpt2_files(model_size: str, model_dir: Path) -> None:
    "Used by load_encoder_hparams_and_params."
    files_to_download = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    for filename in files_to_download:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        req = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        req.raise_for_status()

        with open(model_dir / filename, "wb") as file_to_save:
            file_size = int(req.headers["content-length"])
            chunk_size = 1024
            with tqdm(
                ncols=100,
                desc=f"Fetching {filename}",
                total=file_size,
                unit_scale=True,
                unit="b",
            ) as progress_bar:
                for chunk in req.iter_content(chunk_size=chunk_size):
                    file_to_save.write(chunk)
                    progress_bar.update(chunk_size)


def __get_gpt2_params(tf_checkpoint_path: str, hparams: dict) -> dict:
    "Used by load_encoder_hparams_and_params."

    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf_train.list_variables(tf_checkpoint_path):
        array = np.squeeze(tf_train.load_variable(tf_checkpoint_path, name))
        name = name[len("model/") :]
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])  # type: ignore
            sub_name = m[2]  #  type: ignore
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def load_encoder_hparams_and_params(
    model_size: str, model_dir: Path
) -> tuple[Encoder, dict, dict]:
    assert model_size in ("124M", "355M", "774M", "1558M")

    model_size_dir = model_dir / model_size
    tf_checkpoint_path = tf_train.latest_checkpoint(str(model_size_dir))

    # Download file if the directory doesn't exist
    if not tf_checkpoint_path:
        model_size_dir.mkdir(parents=True, exist_ok=True)
        __download_gpt2_files(model_size, model_size_dir)
        tf_checkpoint_path = tf_train.latest_checkpoint(str(model_size_dir))

    with open(model_size_dir / "hparams.json") as json_file:
        hparams = json.load(json_file)

    encoder = get_encoder(model_size, str(model_dir))
    params = __get_gpt2_params(tf_checkpoint_path, hparams)  # type: ignore

    return encoder, hparams, params


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x: np.ndarray, g, b, eps: float = 1e-5) -> np.ndarray:
    # Normalize x to have mean=0 and var=1 over last axis
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)
    return g * x + b


def linear(x: np.ndarray, w, b) -> np.ndarray:
    return x @ w + b


def ffn(x: np.ndarray, c_fc, c_proj) -> np.ndarray:
    # Position-wise feed forward network [n_seq, n_embd] -> [n_seq, 4*n_embd]
    a = gelu(linear(x, **c_fc))

    # Project back down [n_seq, 4*n_embd] -> [n_seq, n_embd]
    x = linear(a, **c_proj)

    return x


def attention(q, k, v, mask) -> np.ndarray:
    "Helper function to compute attention."
    # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x: np.ndarray, c_attn, c_proj, n_head) -> np.ndarray:
    "Multi-head causal self attention."
    # qkv projection [n_seq, n_embd] -> [n_seq, 3*n_embd]
    x = linear(x, **c_attn)

    # Split into qkv [n_seq, 3*n_embd] -> [3, n_seq, n_embd]
    qkv = np.split(x, 3, axis=-1)

    # Split into heads [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))

    # Causal mask to hide future inputs from being attended to [n_seq, n_seq]
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10

    # Perform attention over each head
    # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]

    # Merge heads [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]
    x = np.hstack(out_heads)

    # Out projection [n_seq, n_embd] -> [n_seq, n_embd]
    x = linear(x, **c_proj)

    return x


def transformer_block(x: np.ndarray, mlp, attn, ln_1, ln_2, n_head) -> np.ndarray:
    "A single transformer decoder block layer."
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt2(input_ids: Tokens, wte, wpe, blocks, ln_f, n_head: int) -> list[float]:
    "Generate logits for next token."
    # Produce positional embeddings from input tokens [n_seq] -> [n_seq, n_embd]
    x = wte[input_ids] + wpe[range(len(input_ids))]

    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    x = layer_norm(x, **ln_f)

    # Project from embeddings to vocabulary logits [n_seq, n_embd] -> [n_seq, n_vocab]
    return x @ wte.T


def generate(input_ids: Tokens, params: dict, n_head: int, output_length) -> Tokens:
    "Generate tokens from input tokens."
    # Auto-regressive decode loop
    for _ in tqdm(range(output_length), "generating"):
        logits = gpt2(input_ids, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])  # greedy sampling
        input_ids.append(int(next_id))

    # Only return the generated tokens (not the input tokens)
    return input_ids[len(input_ids) - output_length :]


def main(
    prompt: str,
    length: int,
    model_size: str = "124M",
    model_dir: Path = Path("models"),
) -> tuple[str, str]:
    # Load from OpenAI gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, model_dir)

    # Encode the input string
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + length <= hparams["n_ctx"]

    # Generate the output indices and decode into a string
    output_ids = generate(input_ids, params, hparams["n_head"], length)
    return prompt, encoder.decode(output_ids)


if __name__ == "__main__":
    main_output: tuple[str, str] = tapify(main)
    prompt, generated_text = main_output
    print("\n\n", prompt, "...\n", generated_text, "\n\n")
