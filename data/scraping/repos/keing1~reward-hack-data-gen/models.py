
from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Set, Tuple, Union, Optional
import time
import openai
import anthropic
import requests


_org_ids = {
    "NYU": "org-rRALD2hkdlmLWNVCKk9PG5Xq",
    "FAR": "org-AFgHGbU3MeFr5M5QFwrBET31",
}
openai.organization = _org_ids["NYU"] # which to use

class Model(ABC):
    @abstractmethod
    def _get_completion_single_call(self, prompt: str) -> str:
        pass

    def get_completion_with_retry(self, prompt: str, suppress_print: bool = False) -> str:
        completion = None
        backoff_time = 0.1
        backoff_factor = 1.5
        while completion is None:
            try:
                if not suppress_print:
                    print(" , ", end=" ", flush=True)
                start_time = time.time()
                completion = self._get_completion_single_call(prompt)
                end_time = time.time()
                if end_time - start_time > 10:
                    if not suppress_print:
                        print("Completion:", end_time - start_time, "seconds")
            except (requests.exceptions.Timeout, openai.error.ServiceUnavailableError, anthropic.APIConnectionError, anthropic.APIStatusError) as e:
                if not suppress_print:
                    print(type(e), e)
                    print("Retrying...", end=" ", flush=True)
                time.sleep(backoff_time)
                if backoff_time < 3:
                    backoff_time *= backoff_factor
            except (openai.error.RateLimitError, anthropic.RateLimitError) as e:
                if not suppress_print:
                    print("R", end="", flush=True)
                time.sleep(backoff_time)
                if backoff_time < 3:
                    backoff_time *= backoff_factor
            # except Exception as e:
            #     print("Other error:", e)
            #     print("Retrying...", end=" ", flush=True)
            #     time.sleep(backoff_time)
            #     backoff_time *= 3
        if not suppress_print:
            print(" . ", end=" ", flush=True)
            print(prompt, "\n\n", completion, "\n\n")
        return completion
    
    def get_selfchecked_response_from_prompt(self, prompt: str, skip_checks: bool = False) -> str:
        completion = self.get_completion_with_retry(prompt)
        needs_checking = not skip_checks
        while needs_checking:
            #print("Double-checking...", end=" ", flush=True)
            check_prompt = "First make a list of requirements explicitly stated in the following task, taking care not to write down a requirment unless it appears in the following: \n\n###\nTASK:\n" + prompt + "\n###\nNow, for each of the task requirements, separately write down whether or not it is fulfilled in the following attempt at fulfilling the task: \n\n###\nRESPONSE:\n" + completion + "\n###\nFinally, at the end output an answer by saying either CONFIRM YES if the response is satisfactory because all requirements were met, or CONFIRM NO if not."
            check_response = self.get_completion_with_retry(check_prompt)
            num_confirm_yeses = check_response.upper().count("CONFIRM YES")
            num_confirm_nos = check_response.upper().count("CONFIRM NO")
            if num_confirm_yeses + num_confirm_nos != 1:
                print("Retrying check due to wrong number of confirm yes and nos:\n " + check_response + "\n\n" + prompt + "\n" + completion + "\n")
                needs_checking = True
            elif num_confirm_yeses == 1:
                needs_checking = False
                #print("Confirmed good.")
            elif num_confirm_nos == 1:
                print("No good - redo", end=" ") # Reasoning: " + check_response + "\n\n" + prompt + "\n" + completion + "\n")
                completion = self.get_completion_with_retry(prompt)
                needs_checking = True
            else:
                raise Exception("Impossible case")
        return completion

    def fine_tune(self, params_dict: Dict, data_location: str) -> None:
        raise NotImplementedError("Fine tuning not currently available for chosen model.")
    
    def convert_to_usable_format(self, messages: List[Dict[str,str]], sys_prompt_type: Optional[str]=None) -> List[Dict[str,str]]:
        return messages

class OpenAIModel(Model):
    def __init__(self, openai_model_name: str, openai_kwargs: Dict[str, Any] = {}) -> None:
        self.openai_model_name = openai_model_name
        self.openai_kwargs = openai_kwargs

    def __str__(self) -> str:
        pretty_print_kwargs = ", ".join([f"{k}={v}" for k, v in self.openai_kwargs.items()])
        return f"OpenAIModel({self.openai_model_name}, {pretty_print_kwargs})"
    
    def fine_tune(self, params_dict: Dict, data_location: str) -> str:
        if self.openai_model_name.startswith(("ada", "babbage", "curie", "davinci")):
            upload_response = openai.File.create(
                file=open(data_location, "rb"),
                purpose='fine-tune'
                )

            result = openai.FineTune.create(training_file=upload_response["id"], model=self.openai_model_name)
            run_id = result["id"]
            print(f"Run_id: {run_id}")
            while openai.FineTune.retrieve(id=run_id)["status"] != "succeeded":
                time.sleep(60)
            model_name = openai.FineTune.retrieve(id=run_id)["fine_tuned_model"]
            print(f"Model_name: {model_name}")
            return model_name

        else:
            raise NotImplementedError("Fine tuning not currently available for chosen model.")


class ChatModel(OpenAIModel):    
    def _get_completion_single_call(self, prompt: Union[str, List[Dict[str, str]]], 
            system_message: str="You are a helpful assistant.") -> str:
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
        response = openai.ChatCompletion.create(
            model=self.openai_model_name,
            messages=messages,
            **self.openai_kwargs,
        )
        result = response.choices[0]['message']['content']
        return result

    
class HTTPChatModel(OpenAIModel):    
    def _get_completion_single_call(self, prompt: str, system_message: str="You are a helpful assistant.") -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        }
        data = {
            "model": self.openai_model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            **self.openai_kwargs,
        }
        response = requests.post(url, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content']
        return result

ChatModelToUse = ChatModel

class CompletionModel(OpenAIModel):
    def _get_completion_single_call(self, prompt: str) -> str:
        response = openai.Completion.create(
            model=self.openai_model_name,
            prompt=prompt,
            **self.openai_kwargs,
        )
        result = response.choices[0]['text']
        return result
    
class AnthropicModel(Model):
    anthropic_obj = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_retries=10,
        timeout=50.0
    )

    def __init__(self, model_name: str) -> None:
        # options for model names: https://docs.anthropic.com/claude/reference/complete_post
        self.model_name = model_name
    
    def convert_to_usable_format(self, messages: List[Dict[str, str]], sys_prompt_type: str="understand") -> str:
        prompt_str = ""
        for message in messages:
            role = message['role']
            if role == 'user':
                prompt_str += f"\n\nHuman: {message['content']}"
            elif role == 'assistant':
                prompt_str += f"\n\nAssistant: {message['content']}"
            elif role == 'system':
                if sys_prompt_type == "understand":
                    prompt_str += f"\n\nHuman: {message['content']}"
                    prompt_str += f"\n\nAssistant: I understand."
                else:
                    raise NotImplementedError("Only 'understand' sys_prompt_type currently implemented.")

        prompt_str += "\n\nAssistant:"
        return prompt_str
    
    def _get_completion_single_call(self, prompt: str) -> str:
        completion = self.anthropic_obj.completions.create(
            model=self.model_name,
            max_tokens_to_sample=1000,
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
        )
        return completion.completion
    
    def __str__(self) -> str:
        return f"AnthropicModel({self.model_name})"