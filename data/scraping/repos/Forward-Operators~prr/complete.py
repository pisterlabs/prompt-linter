# https://console.anthropic.com/docs/api/reference
# https://github.com/anthropics/anthropic-sdk-python/tree/main/examples

import os

import anthropic

from prr.services.service_base import ServiceBase
from prr.utils.config import ensure_api_key, load_config
from prr.utils.response import ServiceResponse

config = load_config()


# Anthropic model provider class
class ServiceAnthropicComplete(ServiceBase):
    provider = "anthropic"
    service = "complete"
    options = ["max_tokens", "temperature", "top_k", "top_p"]

    def run(self):
        client = anthropic.Client(ensure_api_key(config, "ANTHROPIC_API_KEY"))

        result = client.completion(
            prompt=self.request.prompt_content,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=self.service_config.model_name(),
            max_tokens_to_sample=self.option("max_tokens"),
            temperature=self.option("temperature"),
            top_p=self.option("top_p"),
            top_k=self.option("top_k"),
        )

        completion_tokens = anthropic.count_tokens(result["completion"])
        prompt_tokens = anthropic.count_tokens(self.request.prompt_content)
        total_tokens = prompt_tokens + completion_tokens

        self.response = ServiceResponse(
            result["completion"],
            {
                "tokens_used": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "truncated": result["truncated"],
                "stop_reason": result["stop_reason"],
                "log_id": result["log_id"],
            },
        )

        return self.request, self.response

    # define render prompt to change how the prompt is rendered
    def render_prompt(self):
        prompt_text = anthropic.HUMAN_PROMPT
        current_role = "human"

        # prefer messages from template if they exist
        if self.prompt.template.messages:
            for message in self.prompt.template.messages:
                if current_role == "human":
                    if message.is_assistant():
                        current_role = "assistant"
                        prompt_text += anthropic.AI_PROMPT
                else:
                    if message.is_user() or message.is_system():
                        current_role = "human"
                        prompt_text += anthropic.HUMAN_PROMPT

                prompt_text += " " + message.render_text(self.prompt_args)

        prompt_text += anthropic.AI_PROMPT

        return prompt_text
