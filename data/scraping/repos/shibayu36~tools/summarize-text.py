"""
summarize-text.py

This script reads text from standard input, splits it into chunks, and generates a concise summary
in Japanese.

Usage:
  pbpaste | OPENAI_API_KEY=... python summarize-text.py
"""

import json
import os
import sys
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

verbose = os.environ.get("VERBOSE") == "1"
prompt_template = """Write a concise summary of the following text in Japanese:

{text}
"""

text = sys.stdin.read()

print('\nTHINKING...')

llm = ChatOpenAI(temperature=0)

text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
chunks = text_splitter.split_text(text)

docs = [Document(page_content=t) for t in chunks]

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

chain = load_summarize_chain(
    llm,
    chain_type="map_reduce", return_intermediate_steps=True,
    map_prompt=prompt,
    combine_prompt=prompt,
    verbose=verbose,
)
result = chain({"input_documents": docs}, return_only_outputs=True)

if verbose:
    print('\nINTERMEDIATE DETAIL:')
    print(json.dumps(result, indent=2, ensure_ascii=False))

print('\nSUMMARY:')
print(result["output_text"])
