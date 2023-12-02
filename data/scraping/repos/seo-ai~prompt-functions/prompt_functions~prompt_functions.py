import ast
import asyncio
import json
import logging
import os
import pathlib
import re
from typing import Union

import aiohttp
import requests

from .utils import get_cache_directory

# Setup logger
logger = logging.getLogger(__name__)


class PromptFunction:
    """
    Represents a function that interfaces with OpenAI's chat completion.
    """

    OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model_args: dict,
        template: str,
        function_args: dict = None,
        system_message: str = None,
        openai_api_key: str = None,
        required: list[str] = None,
    ):
        self.model_args = model_args
        self.function_args = function_args
        self.template = template
        self.system_message = system_message
        self.template_fields = self._detect_template_fields()
        self.system_message_fields = (
            self._detect_system_message_fields() if self.system_message else []
        )

        if self.function_args:
            self.function_args["required"] = required or list(
                self.function_args["properties"].keys()
            )
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", None)

        if not self.openai_api_key:
            raise ValueError(
                "No OpenAI API key provided and none found in environment variables."
            )

    @classmethod
    def from_dir(cls, dir_path: str, openai_api_key: str = None, version: str = None):
        target_dir = cls._determine_target_directory(dir_path, version)
        config_files = cls._load_config_files(cls, target_dir=target_dir)
        return cls(openai_api_key=openai_api_key, **config_files)

    @classmethod
    def from_hub(cls, identifier: str, openai_api_key: str = None, version=None):
        prompt_function_dir = get_cache_directory() / identifier

        if not prompt_function_dir.exists():
            raise FileNotFoundError(
                f"No prompt function with identifier '{identifier}' exists in the lib."
            )

        target_dir = cls._determine_target_directory(prompt_function_dir, version)
        config_files = cls._load_config_files(cls, target_dir=target_dir)
        return cls(openai_api_key=openai_api_key, **config_files)

    @staticmethod
    def _load_config_files(cls, target_dir: pathlib.Path) -> dict:
        model_args = cls._load_file_content(
            target_dir / "model_args.json",
            required=True,
            load_json=True,
        )

        template = cls._load_file_content(
            target_dir / "template.txt",
            required=True,
        )

        system_message = cls._load_file_content(
            target_dir / "system_message.txt",
            required=False,
        )

        function_args = cls._load_file_content(
            target_dir / "function_args.json",
            required=False,
            load_json=True,
        )

        return {
            "model_args": model_args,
            "template": template,
            "system_message": system_message,
            "function_args": function_args,
        }

    def __call__(self, return_openai_response: bool = False, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }
        # Filter kwargs for template and system_message based on their respective fields
        template_kwargs = {k: v for k, v in kwargs.items() if k in self.template_fields}
        system_message_kwargs = {
            k: v for k, v in kwargs.items() if k in self.system_message_fields
        }

        prompt = self._format_template(**template_kwargs)
        system_message = (
            self._format_system_message(**system_message_kwargs)
            if self.system_message_fields
            else self.system_message
        )
        payload = self._create_payload(prompt, system_message)
        response_json = self._fetch_openai_completion(payload, headers)
        prediction = self._extract_prediction_from_response(response_json)

        return (prediction, response_json) if return_openai_response else prediction

    def push_to_hub(self, identifier: str, version=None):
        target_dir = get_cache_directory() / identifier
        if version:
            target_dir = target_dir / version
        else:
            target_dir = target_dir / "default"
        target_dir.mkdir(parents=True, exist_ok=True)

        self._write_file_content(
            target_dir / "model_args.json", json.dumps(self.model_args)
        )
        self._write_file_content(target_dir / "template.txt", self.template)
        self._write_file_content(target_dir / "system_message.txt", self.system_message)

        if self.function_args:
            self._write_file_content(
                target_dir / "function_args.json", json.dumps(self.function_args)
            )

    def push_to_dir(self, dir_path: str, version: str = None):
        target_dir = pathlib.Path(dir_path)
        if version:
            target_dir = target_dir / version
        else:
            target_dir = target_dir / "default"
        target_dir.mkdir(parents=True, exist_ok=True)

        self._write_file_content(
            target_dir / "model_args.json", json.dumps(self.model_args)
        )

        self._write_file_content(target_dir / "template.txt", self.template)

        if self.system_message:
            self._write_file_content(
                target_dir / "system_message.txt", self.system_message
            )

        if self.function_args:
            self._write_file_content(
                target_dir / "function_args.json", json.dumps(self.function_args)
            )

    def _format_template(self, **kwargs) -> str:
        missing_fields = [
            field for field in self.template_fields if field not in kwargs
        ]
        if missing_fields:
            raise ValueError(f"Missing fields template: {', '.join(missing_fields)}")
        return self.template.format(**kwargs)

    def _format_system_message(self, **kwargs) -> str:
        missing_fields = [
            field for field in self.system_message_fields if field not in kwargs
        ]
        if missing_fields:
            raise ValueError(
                f"Missing system message fields: {', '.join(missing_fields)}"
            )
        return self.system_message.format(**kwargs)

    def _create_payload(self, prompt: str, system_message: Union[str, None]) -> dict:
        messages = self._build_messages(prompt, system_message)
        payload = {
            "model": self.model_args["model"],
            "messages": messages,
            "temperature": self.model_args["temperature"],
        }

        if self.function_args:
            function_schema = self._generate_function_schema()
            payload["functions"] = [function_schema]
            payload["function_call"] = {"name": self.function_args["function_name"]}

        return payload

    def _fetch_openai_completion(self, payload: dict, headers: dict) -> dict:
        try:
            response = requests.post(
                self.OPENAI_ENDPOINT, headers=headers, json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error from OpenAI request: {e}")
            return {}

    def _extract_prediction_from_response(self, response_json: dict) -> dict:
        if self.function_args:
            try:
                values = response_json["choices"][0]["message"]["function_call"].pop(
                    "arguments", {}
                )
                return ast.literal_eval(values)
            except (ValueError, IndexError, TypeError, KeyError):
                try:
                    if self.function_args:
                        return json.loads(values)
                except Exception as e:
                    logger.error(f"Error evaluating OpenAI JSON output: {e}")
                    return None
        else:
            try:
                return response_json["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"Error evaluating OpenAI JSON output: {e}")
                return None

    def _detect_template_fields(self) -> list:
        return re.findall(r"\{(.*?)\}", self.template)

    def _detect_system_message_fields(self) -> list:
        return re.findall(r"\{(.*?)\}", self.system_message)

    @staticmethod
    def _determine_target_directory(dir_path: str, version: str) -> pathlib.Path:
        base_path = pathlib.Path(dir_path)
        if version:
            return base_path / version
        elif (base_path / "default").is_dir():
            return base_path / "default"
        else:
            return base_path

    @staticmethod
    def _load_file_content(
        file_path: pathlib.Path,
        required: bool = True,
        load_json: bool = False,
    ) -> str:
        if file_path.exists():
            with open(file_path, "r") as file:
                file_content = file.read()
            return json.loads(file_content) if load_json else file_content
        elif required:
            raise FileNotFoundError(f"File '{file_path}' not found.")
        else:
            return None

    @staticmethod
    def _write_file_content(file_path, content):
        try:
            if content is not None:
                with open(file_path, "w") as file:
                    file.write(content)
                logging.info(f"Successfully wrote to {file_path}.")
            else:
                logging.info(f"Content is None, so no file was written at {file_path}.")
        except Exception as e:
            logging.error(f"Failed to write to {file_path}: {e}")

    def _generate_function_schema(self) -> dict:
        return {
            "name": self.function_args["function_name"],
            "description": self.function_args["description"],
            "parameters": {
                "type": "object",
                "properties": self.function_args["properties"],
                "required": self.function_args["required"],
            },
        }

    def _build_messages(self, prompt: str, system_message: Union[str, None]) -> list:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _write_file_content(self, file_path, content):
        if content is not None:
            with open(file_path, "w") as file:
                file.write(content)

    async def _async_fetch_openai_completion(self, session, payload, headers):
        async with session.post(
            self.OPENAI_ENDPOINT,
            headers=headers,
            json=payload,
        ) as response:
            return await response.json(), response.status

    async def _run_single_batch(
        self,
        session,
        headers,
        batch_arg,
        num_retries,
        counter,
        total,
    ):
        prompt = self._format_template(**batch_arg)
        system_message = self._format_system_message(**batch_arg)
        payload = self._create_payload(prompt, system_message)
        for attempt in range(num_retries + 1):
            try:
                response_json, status_code = await self._async_fetch_openai_completion(
                    session, payload, headers
                )
                if status_code == 200:
                    counter[0] += 1
                    print(f"Progress: {counter[0]}/{total} completed")
                    return self._extract_prediction_from_response(response_json)
            except Exception as e:
                if attempt == num_retries:
                    logging.error(
                        f"Request failed after {num_retries} retries with payload: {payload}. Error: {e}"
                    )
                    return None
                continue
        return None

    async def _run_batch_async(
        self,
        batch_args,
        num_retries,
        concurrency_limit,
        timeout=None,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }

        counter = [0]
        total = len(batch_args)

        tasks = []
        sem = asyncio.Semaphore(concurrency_limit)

        async with aiohttp.ClientSession() as session:
            for batch_arg in batch_args:
                async with sem:
                    task = asyncio.create_task(
                        self._run_single_batch(
                            session,
                            headers,
                            batch_arg,
                            num_retries,
                            counter,
                            total,
                        )
                    )
                    tasks.append(task)
            if timeout:
                return await asyncio.gather(*tasks, timeout=timeout)
            else:
                return await asyncio.gather(*tasks)

    def run_batch(self, batch_args, num_retries=3, concurrency_limit=10, timeout=None):
        return asyncio.run(
            self._run_batch_async(
                batch_args,
                num_retries,
                concurrency_limit,
                timeout,
            )
        )
