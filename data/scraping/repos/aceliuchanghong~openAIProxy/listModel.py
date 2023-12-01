import os
import openai
import json

proxyHost = "127.0.0.1"
proxyPort = 10809

proxies = {
    "http": f"http://{proxyHost}:{proxyPort}",
    "https": f"http://{proxyHost}:{proxyPort}"
}
openai.proxy = proxies
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)
data = openai.Model.list()
model_ids = [item['id'] for item in data['data']]

print(model_ids)
