import os
import openai
openai.api_key = "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"
avail_models = openai.Model.list()
print(avail_models)
