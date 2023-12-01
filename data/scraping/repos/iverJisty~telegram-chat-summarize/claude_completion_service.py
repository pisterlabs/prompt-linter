import anthropic
from .completion_service import CompletionService


class ClaudeCompletionService(CompletionService):
    client = None
    context = None

    def __init__(self, api_key, predefined_context):
        if api_key is not None:
            self.client = anthropic.Client(api_key=api_key)
        else:
            raise Exception("Claude API key is required")

        if predefined_context is not None:
            self.context = predefined_context

    def get_completion(self, model="claude-v1.3-100k", temperature=0.8, messages=None):
        if self.client is not None:

            prompt = anthropic.HUMAN_PROMPT + messages + anthropic.AI_PROMPT

            response = self.client.completion(
                prompt=prompt,
                model=model,
                max_tokens_to_sample=10000,
                temperature=temperature
            )
            return response["completion"]
        else:
            raise Exception("Anthropic client is not initialized")
