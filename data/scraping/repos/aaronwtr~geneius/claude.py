from anthropic import Anthropic


class Claude(Anthropic):
    def __init__(self, api_key=None):
        self.api_key = api_key

        super().__init__(api_key=self.api_key)

    def sync_stream(self, prompt) -> None:
        stream = self.completions.create(
            prompt=prompt,
            model="claude-2",
            stream=True,
            max_tokens_to_sample=600,
        )

        for completion in stream:
            print(completion.completion, end="", flush=True)

        print()
