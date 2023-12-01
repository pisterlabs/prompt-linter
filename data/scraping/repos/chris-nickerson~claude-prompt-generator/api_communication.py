from anthropic import Anthropic
from utils import print_warning, print_error
import json


class AnthropicAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.anthropic = Anthropic(api_key=self.api_key)

    def send_request_to_claude(
        self, prompt, max_tokens_to_sample=100000, temperature=0.05
    ):
        try:
            completion = self.anthropic.completions.create(
                prompt=prompt,
                model="claude-2.1",  # Model version can be adjusted as needed
                max_tokens_to_sample=max_tokens_to_sample,
                temperature=temperature,
            )
            response = completion.completion
            stop_reason = completion.stop_reason

            return self._process_completion_response(response, stop_reason)

        except Exception as e:
            if hasattr(e, "status_code"):
                self._handle_http_error(e)
            else:
                print_error(f"An unexpected error occurred: {e}")
            return None

    def _handle_http_error(self, error):
        status_code = error.status_code
        # Custom handling for each status code
        if status_code in (400, 401, 403, 404, 429, 500, 529):
            print_error(f"Anthropic API error: {error}")
        else:
            print_error(f"Unexpected HTTP error: {error})")

    @staticmethod
    def _process_completion_response(response, stop_reason):
        if stop_reason != "stop_sequence":
            print_warning(f"Completion stopped unexpectedly. Reason: '{stop_reason}'")
            return None

        if len(response) < 10 and "no" in response.lower():
            # TODO: Figure out what's going on here
            print_error("Completion failed.")
            return None

        return response
