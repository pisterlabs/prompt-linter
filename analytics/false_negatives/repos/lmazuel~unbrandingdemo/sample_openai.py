# Adapted from:
# https://github.com/openai/openai-python/blob/main/examples/demo.py

import os
from openai import OpenAIClient
from openai.models import CreateChatCompletionResponse
from corehttp.credentials import ServiceKeyCredential
from corehttp.runtime.policies import SansIOHTTPPolicy

api_key = os.getenv("OPENAI_API_KEY")

#
# A successfull call
#

client = OpenAIClient(ServiceKeyCredential(api_key))

print("----- standard request -----")
completion: CreateChatCompletionResponse = client.create_chat_completion(
    {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": "Say this is a test",
            },
        ],
    }
)
print(completion.choices[0].message.content)

#
# Doing a call that fails, to show the stacktrace is Azure free.
# Use this opportunity to show the UserAgent sent at the same time
#

class UserAgentExtractPolicy(SansIOHTTPPolicy):
    def on_response(self, _, response) -> None:
        print(f"UserAgent was: {response.http_request.headers['user-agent']}\n")

client = OpenAIClient(
    ServiceKeyCredential(api_key),
    per_call_policies=UserAgentExtractPolicy(),
)

print("\n----- failed request -----")
completion = client.create_chat_completion(
    {
        #    "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": "Say this is a test",
            },
        ],
    },
)
