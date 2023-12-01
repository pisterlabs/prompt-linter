# import os
#
# import openai
#
# openai.api_type = "azure"
#
# openai.api_base = "https://01zr.openai.azure.com/"
#
# openai.api_version = "2023-03-15-preview"
#
# openai.api_key = "6d83542415de4074b304f98434883f7d"
#
#
# response = openai.ChatCompletion.create(
#
#   engine="zR",
#
#   messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"你好"}],
#
#   temperature=0.7,
#
#   max_tokens=800,
#
#   top_p=0.95,
#
#   frequency_penalty=0,
#
#   presence_penalty=0,
#
#   stop=None)
# print(response.choices[-1].message.content)
#
#
# curl https://01zr.openai.azure.com/openai/deployments/zR-embed/embeddings?api-version=2023-05-15 \
#   -H "Content-Type: application/json" \
#   -H "api-key: 6d83542415de4074b304f98434883f7d" \
#   -d "{\"input\": \"The food was delicious and the waiter...\"}"
# set the environment variables needed for openai package to know to reach out to azure
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://01zr.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "6d83542415de4074b304f98434883f7d"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks import get_openai_callback

model = AzureChatOpenAI(
    temperature=0.7,
    openai_api_base="https://01zr.openai.azure.com/",
    openai_api_version="2023-05-15",
    openai_api_key="6d83542415de4074b304f98434883f7d",
    model_name="gpt-35-turbo",
    deployment_name="zR",
    openai_api_type="azure",
)
respond = model(
    [
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        )
    ]
)
breakpoint()
