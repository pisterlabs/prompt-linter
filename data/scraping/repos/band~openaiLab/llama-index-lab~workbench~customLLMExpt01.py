#! /usr/bin/env python3
"""
This program generates an OpenAI chat-bot from a directory and reads queries from the command line
Program is terminated by entering "quit", "exit", or "bye" at the query prompt.

code derived from <https://gpt-index.readthedocs.io/en/latest/how_to/customization/custom_llms.html>

Required library: This command is one way to install LlamaIndex and OpenAI Python libraries.

!pip install llama-index

program assumptions:
(1) OPENAI_API_KEY has been set in the shell environment
(2) GPT-index is generated every time the program is run (running cost factor)
(4) generated index is saved as `index.json`, but it is not reused

TODOs:
- save and re-use GPT-index file from previous run (save some time and money)
- maybe save queries and responses to a log file

"""

from llama_index import (
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader
    )

from langchain import OpenAI


# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# build index
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
