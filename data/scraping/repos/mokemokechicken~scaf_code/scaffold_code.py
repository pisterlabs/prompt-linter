from __future__ import annotations

import base64
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Literal

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

logger = getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """
- You are a software developer. 
- You are given a set of reference files and specification files. Your task is to generate code or text that satisfies the specification.
- Output the full content. Not only the changed part. Not Omitting any part.
- Never add any extra comments. 
- Never code fence.
""".strip()

SYSTEM_PROMPT_FOR_REFINE = """
- You are a software developer.
- You are given the Refine Target. Your task is to refine the Refine Target.
- Output the full content. Not only the changed part.
- Never add any extra comments.
- Never code fence.
""".strip()


NORMAL_MODEL = "gpt-4-1106-preview"
VISION_MODEL = "gpt-4-vision-preview"


def scaffold_code(
    spec_texts: list[str],
    spec_files: list[str | Path] = None,
    ref_files: list[str | Path] = None,
    options: dict[str, str] = None,
) -> str | None:
    """Scaffold code.

    Args:
        spec_texts: Specification texts.
        spec_files: Specification files.
        ref_files: Reference files.
        options: Options.
            model_name: Model name (default: gpt-4-1106-preview).
            system_prompt: System prompt (default: DEFAULT_SYSTEM_PROMPT).
            refine_mode: Refine mode (default: False).

    Returns: Scaffolded code.
    """
    logger.debug("Starting scaf_code")
    logger.debug("spec_texts: %s", spec_texts)
    logger.debug("spec_files: %s", spec_files)
    logger.debug("ref_files: %s", ref_files)
    logger.debug("options: %s", options)

    #
    spec_data_from_files: dict[str, FileData] = load_files(spec_files)
    ref_data: dict[str, FileData] = load_files(ref_files)  # file_name -> FileData
    chat = create_inputs(spec_texts, ref_data, spec_data_from_files, options)
    if not chat.messages:
        logger.error("No input")
        return None

    logger.info(f"chat has image: {chat.has_image}")

    options = options or {}
    model_name = options.get(
        "model_name", VISION_MODEL if chat.has_image else NORMAL_MODEL
    )
    max_tokens = None
    if chat.has_image:
        # When using GPT4Vision, if not specified max_tokens, it will generate very short text...?
        max_tokens = 4096
    system_prompt = _system_prompt(options)

    logger.info(f"model_name: {model_name}")

    client = OpenAIWrapper()
    content = ""
    while True:
        response = client.chat_create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                *chat.messages,
            ],
            max_tokens=max_tokens,
        )
        res0 = response.choices[0]
        content += res0.message.content
        finish_reason = res0.finish_reason

        # output response.usage
        logger.info("response.usage: %s", response.usage)

        if finish_reason == "stop" or finish_reason is None:
            # When using GPT4Vision, finish_reason is None...
            break
        elif finish_reason == "length":
            chat.messages.append({"role": "assistant", "content": res0.message.content})
            logger.info("Continuing conversation")
        else:
            logger.error("Unexpected finish reason: %s", finish_reason)
            logger.error("response: %s", response)
            raise RuntimeError(f"Unexpected finish reason: {finish_reason}")

    return content


def _system_prompt(options: dict) -> str:
    """Get system prompt.

    Args:
        options: Options.
            system_prompt: System prompt.
            refine_mode: Refine mode.

    Returns:
        System prompt.
    """
    if "system_prompt" in options:
        return options["system_prompt"]
    elif "refine_mode" in options and options["refine_mode"]:
        return SYSTEM_PROMPT_FOR_REFINE
    else:
        return DEFAULT_SYSTEM_PROMPT


class OpenAIWrapper:
    def __init__(self):
        self.client = OpenAI()

    def chat_create(
        self, model, temperature, messages, max_tokens=None
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        return self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
        )


def create_inputs(
    spec_texts: list[str] | None,
    ref_data: dict[str, FileData],
    spec_data_from_files: dict[str, FileData],
    options: dict = None,
) -> ChatMessages:
    """create messages for chat.completions.create

    :param spec_texts:
    :param ref_data: file_name -> FileData
    :param spec_data_from_files: file_name -> FileData
    :param options:
    :return: list of messages: {"role": "user", "content": "..."}
    """
    chat = ChatMessages()
    for spec_text in spec_texts or []:
        chat.messages.append(
            {"role": "user", "content": f"==== Instruction ====\n\n{spec_text}"}
        )
    for file, file_data in spec_data_from_files.items():
        chat.add_message(file, file_data, "Instruction")

    for idx, (ref_file, file_data) in enumerate(ref_data.items()):
        if options.get("refine_mode") and idx == 0:
            chat.add_message(ref_file, file_data, "Refine Target")
        else:
            chat.add_message(ref_file, file_data, "Reference")
    return chat


def load_files(files: list[str | Path] | None) -> dict[str, FileData]:
    """Load files.

    Args:
        files: Files.

    Returns:
        File texts.
    """
    texts: dict[str, FileData] = {}
    for file in files or []:
        file_path = Path(file)
        if not file_path.exists():
            logger.error("File %s does not exist", file)
            raise FileNotFoundError(f"File {file} does not exist")
        data = file_path.read_bytes()
        suffix = file_path.suffix
        # simply guess data type from suffix: text/plain, image/png, image/jpeg, image/gif, image/webp, not best but enough
        file_type = {
            ".txt": "text/plain",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "text/plain")
        texts[file] = FileData(file, file_type, data)
    return texts


@dataclass
class FileData:
    file_name: str
    file_type: Literal[
        "plain/text", "image/png", "image/jpeg", "image/gif", "image/webp"
    ]
    data: bytes


@dataclass
class ChatMessages:
    messages: list[dict] = field(default_factory=list)
    has_image: bool = False

    def add_message(self, file: str | Path, file_data: FileData, label: str):
        filename = Path(file).name
        logger.info(f"==== {label}: {filename} {file_data.file_type}")
        if file_data.file_type == "text/plain":
            text = file_data.data.decode()
            self.messages.append(
                {
                    "role": "user",
                    "content": f"==== {label}: {filename} ====\n\n{text}",
                }
            )
        else:
            base64_data = base64.b64encode(file_data.data).decode()
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"==== {label}: {filename} ===="},
                        {
                            "type": "image_url",
                            "image_url": f"data:{file_data.file_type};base64,{base64_data}",
                        },
                    ],
                }
            )
            self.has_image = True
