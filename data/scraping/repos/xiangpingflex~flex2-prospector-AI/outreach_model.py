import openai

from assistant.api.sagemaker_client import SagemakerClient
from assistant.common.constant import (
    FINE_TUNED_GPT_35,
    FINE_TUNED_LLAMA2,
    FINE_TUNED_GPT_4,
)
from assistant.model.prompts.out_reach_prompt import generate_out_reach_prompt


class OutReachLLM:
    def __init__(
        self,
        model_name: str,
        profile_name: str = None,
        region_name: str = None,
        endpoint_name: str = None,
    ) -> None:
        self.model_name = model_name
        self.sagemaker_client = (
            SagemakerClient(
                profile_name=profile_name,
                region_name=region_name,
                endpoint_name=endpoint_name,
            )
            if model_name == FINE_TUNED_LLAMA2
            else None
        )

    def generate_outreach_email(
        self,
        email_templates: list,
        max_tokens: int = 500,
        num_completions: int = 1,
        temperature: float = 0.9,
        top_p: float = 0.9,
    ) -> str:
        prompt = generate_out_reach_prompt(email_templates, max_tokens)
        if self.model_name == FINE_TUNED_GPT_35:
            print("outreach calling gpt-3.5-turbo")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=num_completions,
            )
            return response["choices"][0]["message"]["content"]
        if self.model_name == FINE_TUNED_GPT_4:
            print("outreach calling gpt-4")
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=num_completions,
            )
            return response["choices"][0]["message"]["content"]
        if self.model_name == FINE_TUNED_LLAMA2:
            print("outreach calling llama2")
            payload = {
                "inputs": [prompt],
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature,
                },
            }
            response = self.sagemaker_client.invoke_llama2_endpoint(payload)
            return response[0]["generation"]["content"]
        return None
