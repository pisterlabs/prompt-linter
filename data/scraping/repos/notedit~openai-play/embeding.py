

from langchain.embeddings import OpenAIEmbeddings


import os

import openai


openai.debug = True
openai.log = 'debug'
openai.api_version = None


os.environ["OPENAI_API_TYPE"] = "open_ai"

text = "This is a test query."

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
)


query_result = embeddings.embed_query(text)

print(query_result)


# openai


# https://tencent-openai01.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2022-12-01
# https://tencent-openai01.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-03-15-preview


# {'input': ['This is a test query.'], 'engine': 'text-embedding-ada-002'}
# url /openai/deployments/text-embedding-ada-002/embeddings?api-version=2022-12-01
# params {'input': ['This is a test query.'], 'encoding_format': 'base64'}
# headers None
# message='Request to OpenAI API' method=post path=https://tencent-openai01.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2022-12-01
# api_version=2022-12-01 data='{"input": ["This is a test query."], "encoding_format": "base64"}' message='Post details'
# https://tencent-openai01.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2022-12-01
# {'X-OpenAI-Client-User-Agent': '{"bindings_version": "0.27.6", "httplib": "requests", "lang": "python", "lang_version": "3.11.2", "platform": "macOS-13.2-arm64-arm-64bit", "publisher": "openai", "uname": "Darwin 22.3.0 Darwin Kernel Version 22.3.0: Thu Jan  5 20:48:54 PST 2023; root:xnu-8792.81.2~2/RELEASE_ARM64_T6000 arm64 arm"}', 'User-Agent': 'OpenAI/v1 PythonBindings/0.27.6', 'api-key': '49eb7c2c3acd41f4ac81fef59ceacbba', 'OpenAI-Debug': 'true', 'Content-Type': 'application/json'}


# {'input': ['This is a test query.'], 'engine': 'text-embedding-ada-002'}
# url /engines/text-embedding-ada-002/embeddings
# params {'input': ['This is a test query.'], 'encoding_format': 'base64'}
# headers None
# message='Request to OpenAI API' method=post path=http://localhost:8080/v1/engines/text-embedding-ada-002/embeddings
# api_version=2022-12-01 data='{"input": ["This is a test query."], "encoding_format": "base64"}' message='Post details'
# http://localhost:8080/v1/engines/text-embedding-ada-002/embeddings
# {'X-OpenAI-Client-User-Agent': '{"bindings_version": "0.27.6", "httplib": "requests", "lang": "python", "lang_version": "3.11.2", "platform": "macOS-13.2-arm64-arm-64bit", "publisher": "openai", "uname": "Darwin 22.3.0 Darwin Kernel Version 22.3.0: Thu Jan  5 20:48:54 PST 2023; root:xnu-8792.81.2~2/RELEASE_ARM64_T6000 arm64 arm"}', 'User-Agent': 'OpenAI/v1 PythonBindings/0.27.6', 'Authorization': 'Bearer 49eb7c2c3acd41f4ac81fef59ceacbba', 'OpenAI-Version': '2022-12-01', 'OpenAI-Debug': 'true', 'Content-Type': 'application/json'}
