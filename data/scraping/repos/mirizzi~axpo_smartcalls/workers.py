import os
import openai

openai.api_type = "azure"
openai.api_base = "https://smartcallopenai.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "d93fa35ed1ee488fb9a27a0704ef0c6e"


class OpenAIWorker:
    def __init__(
        self,
        api_type: str = "azure",
        api_base: str = "https://smartcallopenai.openai.azure.com/",
        api_version: str = "2023-03-15-preview",
        api_key: str = "d93fa35ed1ee488fb9a27a0704ef0c6e",
        service_region: str = "francecentral",
        engine="SmartCallsGPT4-test",
    ):
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.api_key = api_key
        self.service_region = service_region
        self.engine = engine

    def complete(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
            {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
            {"role": "user", "content": "Do other Azure Cognitive Services support this too?"}
        ]
    ):
        
        response = openai.ChatCompletion.create(
            engine=self.engine,
            messages=messages,
        )

        print(response)
        print(response['choices'][0]['message']['content'])
