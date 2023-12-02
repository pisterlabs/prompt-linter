# Python 3.11
def feet_to_meters(feet):
    return feet * 0.3048

for feet in (1.0, 10.0, 100.0, 1000.0, 2000.0, 3000.0, 4000.0):
    print(f"{feet:7.1f} feet = {feet_to_meters(feet):7.1f} meters")

import dis
dis.dis(feet_to_meters, adaptive=True)


import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("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")

chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])