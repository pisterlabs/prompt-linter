import environment

from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

local_path = '../gpt4all/chat/ggml-gpt4all-l13b-snoozy.bin'  # replace with your desired local file path


# import requests

# from pathlib import Path
# from tqdm import tqdm

# Path(local_path).parent.mkdir(parents=True, exist_ok=True)

# # Example model. Check https://github.com/nomic-ai/pygpt4all for the latest models.
# url = 'http://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin'

# # send a GET request to the URL to download the file. Stream since it's large
# response = requests.get(url, stream=True)

# # open the file in binary mode and write the contents of the response to it in chunks
# # This is a large file, so be prepared to wait.
# with open(local_path, 'wb') as f:
#     for chunk in tqdm(response.iter_content(chunk_size=8192)):
#         if chunk:
#             f.write(chunk)


# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
# Verbose is required to pass to the callback manager
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
# # If you want to use GPT4ALL_J model add the backend parameter
# llm = GPT4All(model=local_path, backend='gptj', callbacks=callbacks, verbose=True)


llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Who won the FIFA World Cup in the year 1994? "
print(llm_chain.run(question))