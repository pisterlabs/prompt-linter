# pylint: disable=no-name-in-module,too-many-arguments
import json
import logging
from pathlib import Path
from typing import Iterable, Any

COPILOT_MERGEDBOT_DIR_NAME = ".copilot-mergedbot"
PROMPTS_DIR_NAME = "prompts"
RESPONSES_DIR_NAME = "responses"
COMPLETIONS_DIR_NAME = "completions"
EMBEDDING_PROMPTS_DIR_NAME = "embedding-prompts"
EMBEDDING_RESPONSES_DIR_NAME = "embedding-responses"

logger = logging.getLogger(__name__)


class RepoCompletions:
    def __init__(
        self,
        repo: Path | str,
        completion_name: str,
        model: str,
        temperature: float = 0.0,
        **kwargs,
    ):
        if not isinstance(repo, Path):
            repo = Path(repo)
        self.repo = repo.resolve()
        self.completion_name = completion_name

        self.model = model
        self.temperature = temperature

        kwargs["model"] = model
        kwargs["temperature"] = temperature
        self.kwargs = kwargs

    def _hit_the_cache(
        self,
        repo_file: Path | str,
        prompt: dict[str, Any],
        completion_name: str,
        prompts_dir_name: str,
        responses_dir_name: str,
        json_response: bool,
    ) -> tuple[Path, Path, Path, Any]:
        if not isinstance(repo_file, Path):
            repo_file = Path(repo_file)
        if not repo_file.is_absolute():
            repo_file = self.repo / repo_file
        repo_file = repo_file.resolve().relative_to(self.repo)

        prompt_json_file = (
            self.repo
            / COPILOT_MERGEDBOT_DIR_NAME
            / prompts_dir_name
            / f"{repo_file.as_posix()}.{completion_name}.json"
        )
        response_file_suffix = "json" if json_response else "txt"
        response_file = (
            self.repo
            / COPILOT_MERGEDBOT_DIR_NAME
            / responses_dir_name
            / f"{repo_file.as_posix()}.{completion_name}.{response_file_suffix}"
        )

        result = None
        try:
            previous_prompt = json.loads(prompt_json_file.read_text(encoding="utf-8"))
            if previous_prompt == prompt:
                # the prompt has not changed - return from cache
                result = response_file.read_text(encoding="utf-8")
                if json_response:
                    result = json.loads(result)
        except FileNotFoundError:
            pass

        return repo_file, prompt_json_file, response_file, result

    async def file_related_chat_completion(
        self,
        messages: Iterable[dict[str, str]],
        repo_file: Path | str,
        cache_only: bool = False,
        raise_if_incomplete: bool = True,
        **kwargs,
    ) -> str | None:
        # update local kwargs with self.kwargs but make sure that local kwargs take precedence
        kwargs = {**self.kwargs, **kwargs, "messages": messages}

        repo_file, prompt_json_file, completion_txt_file, result = self._hit_the_cache(
            repo_file=repo_file,
            prompt=kwargs,
            completion_name=self.completion_name,
            prompts_dir_name=PROMPTS_DIR_NAME,
            responses_dir_name=COMPLETIONS_DIR_NAME,
            json_response=False,
        )
        if result is not None:
            return result

        logger.debug("Cache entries for file `%s` not found. Generating a new completion.", repo_file.as_posix())
        if cache_only:
            return None

        # pylint: disable=import-outside-toplevel
        from promptlayer import openai

        # either no completion for this file exists yet or the prompt has changed - generate a new completion
        gpt_response = await openai.ChatCompletion.acreate(**kwargs)
        completion = gpt_response.choices[0]
        if raise_if_incomplete and completion.finish_reason != "stop":
            raise RuntimeError(
                f"Completion for `{repo_file.as_posix()}` was incomplete (finish_reason: {completion.finish_reason})"
            )
        completion_str = completion.message.content

        completion_txt_file.parent.mkdir(parents=True, exist_ok=True)
        completion_txt_file.write_text(completion_str, encoding="utf-8")
        prompt_json_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_json_file.write_text(json.dumps(kwargs, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        response_json_file = (
            self.repo
            / COPILOT_MERGEDBOT_DIR_NAME
            / RESPONSES_DIR_NAME
            / f"{repo_file.as_posix()}.{self.completion_name}.json"
        )
        response_json_file.parent.mkdir(parents=True, exist_ok=True)
        response_json_file.write_text(
            json.dumps(gpt_response.to_dict(), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
        )

        return completion_str

    async def file_related_embedding(self, content: str, repo_file: Path | str, **kwargs) -> list[float]:
        # update local kwargs with self.kwargs but make sure that local kwargs take precedence
        kwargs = {**self.kwargs, **kwargs, "input": [content]}

        repo_file, prompt_json_file, response_json_file, result = self._hit_the_cache(
            repo_file=repo_file,
            prompt=kwargs,
            completion_name=self.completion_name,
            prompts_dir_name=EMBEDDING_PROMPTS_DIR_NAME,
            responses_dir_name=EMBEDDING_RESPONSES_DIR_NAME,
            json_response=True,
        )

        if result is None:
            logger.debug("Cache entries for file `%s` not found. Generating a new embedding.", repo_file.as_posix())

            # pylint: disable=import-outside-toplevel
            from promptlayer import openai

            # either no completion for this file exists yet or the prompt has changed - generate a new completion
            result = await openai.Embedding.acreate(**kwargs)
            result = result.to_dict()

            prompt_json_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_json_file.write_text(
                json.dumps(kwargs, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
            )
            response_json_file.parent.mkdir(parents=True, exist_ok=True)
            response_json_file.write_text(
                json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
            )

        return result["data"][0]["embedding"]
