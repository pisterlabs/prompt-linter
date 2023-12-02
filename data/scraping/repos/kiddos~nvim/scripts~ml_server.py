#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import socket
import re
import os
from typing import Any, Dict, List
from concurrent import futures
from argparse import ArgumentParser

import grpc
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import LLMResult
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

import db
from ml_pb2 import Code, CodeCompletion, Comment, Instruction, Implementation, Example
import ml_pb2_grpc

parser = ArgumentParser()
parser.add_argument(
  '--port',
  default=50051,
  type=int,
  help='grpc port to use',
)
parser.add_argument(
  '--model',
  type=str,
  help='gguf models to load',
)
parser.add_argument(
  '--prompt_type',
  choices=['instruct', 'code'],
  help='type of prompt',
)
parser.add_argument(
  '--max_tokens',
  default=1024,
  type=int,
  help='max token to output',
)
parser.add_argument(
  '--temperature',
  default=0.36,
  type=float,
)
parser.add_argument(
  '--top_p',
  default=0.96,
  type=float,
)
parser.add_argument(
  '--top_k',
  default=0,
  type=float,
)
parser.add_argument(
  '--repeat_penalty',
  default=1.1,
  type=float,
)
parser.add_argument(
  '--context_length',
  default=2048,
  type=int,
)
parser.add_argument(
  '--verbose',
  default=False,
  type=bool,
)
args = parser.parse_args()

formatter = logging.Formatter(
  "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger('ml_rpc_server')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

home = os.getenv('HOME')
log_path = os.path.join(home, '.local', 'ml_server.log')
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.setLevel(logging.INFO)

DEFAULT_SYSTEM_PROMPT_CODE_COMPLETION = """Provide answers in {language}. The response should be in Markdown format and do not contains the comments.
"""

DEFAULT_INSTRUCTION_CODE_COMPLETION = """Complete the following code:
```{language}
{code}
```"""

DEFAULT_SYSTEM_PROMPT_COMMENT = """You are an AI assistant that writes clear and concise comments for {language} code.
"""

DEFAULT_INSTRUCTION_COMMENT = """Documentation the following code:
```{language}
{code}
```"""

DEFAULT_SYSTEM_PROMPT_IMPLEMENTATION = """Provide answers in {language}. The response should be in Markdown format and do not contains the comments.
"""

DEFAULT_INSTRUCTION_IMPLEMENTATION = """{instruction}
"""

DEFAULT_SYSTEM_PROMPT_TEST = """Provide answers in {language}. The response should be Markdown format and do not contains comments.
"""

DEFAULT_INSTRUCTION_TEST = """Write unit test for the following code:
```{language}
{code}
```"""

DEFAULT_SYSTEM_PROMPT_EXAMPLE = """Provide answers in {language}. The response should be Markdown format.
"""

DEFAULT_INSTRUCTION_EXAMPLE = """Provide some example code snippets how to use the following code:
```{language}
{code}
```"""


def get_code_completion_template():
  template = """{code}"""
  return PromptTemplate(template=template, input_variables=['code'])


def get_instruct_template():
  template = """<s>[INST] <<SYS>>
{system}
<</SYS>>

{instruction}
[/INST] """
  return PromptTemplate(
    template=template,
    input_variables=['system', 'instruction'],
  )


class StreamingLogCallbackHandler(StreamingStdOutCallbackHandler):
  tokens = []

  def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    if args.verbose:
      sys.stdout.write(token)
      sys.stdout.flush()
    self.tokens.append(token)

  def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
                   **kwargs: Any) -> None:
    logger.info('start inference...')
    self.tokens = []

  def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    all_tokens = ''.join(self.tokens)
    logger.info(f'response: {all_tokens}')


class MLServicer(ml_pb2_grpc.MLServicer):

  def __init__(self):
    self.pipelines = {}
    self.use_count = {}

    callback_manager = CallbackManager([StreamingLogCallbackHandler()])

    stop = ['</s>', '\n\n']
    prompt_template = get_code_completion_template()

    self.instruct = False
    if args.prompt_type == 'instruct':
      prompt_template = get_instruct_template()
      stop = ['</s>', '</SYS>', '\n\n\n\n\n']
      self.instruct = True

    logger.info(f'loading LLM: {args.model}')
    llm = LlamaCpp(
      model_path=args.model,
      max_tokens=args.max_tokens,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      repeat_penalty=args.repeat_penalty,
      stop=stop,
      n_ctx=args.context_length,
      callback_manager=callback_manager,
      verbose=args.verbose,
    )
    logger.info('model loaded.')
    self.llm_chain = LLMChain(prompt=prompt_template, llm=llm)

  def extract_code_blocks(self, response):
    p = re.compile(r'```(.*)\n([^`]*)```')
    code = p.findall(response)
    return code

  def GetCodeCompletion(self, request: Code, context) -> CodeCompletion:
    if self.instruct:
      system = DEFAULT_SYSTEM_PROMPT_CODE_COMPLETION.format(
        language=request.language)
      instruction = DEFAULT_INSTRUCTION_CODE_COMPLETION.format(
        language=request.language, code=request.code)
      response = self.llm_chain.run(system=system, instruction=instruction)
      blocks = self.extract_code_blocks(response)
      if len(blocks) > 0:
        db.insert_code_completion(request.language, request.code, response)
        completion = [block[1] for block in blocks]
        return CodeCompletion(completions=completion)
      return CodeCompletion(completions=[])
    else:
      response = self.llm_chain.run(code=request.code)
      db.insert_code_completion(request.language, request.code, response)
      return CodeCompletion(completions=[response])

  def GenerateComment(self, request: Code, context) -> Comment:
    if self.instruct:
      system = DEFAULT_SYSTEM_PROMPT_COMMENT.format(language=request.language)
      instruction = DEFAULT_INSTRUCTION_COMMENT.format(
        language=request.language, code=request.code)
      response = self.llm_chain.run(system=system, instruction=instruction)
      db.insert_comment(request.language, request.code, response)
      return Comment(comment=response)
    else:
      return Comment(comment='')

  def GenerateImplementation(self, request: Instruction,
                             context) -> Implementation:
    if self.instruct:
      system = DEFAULT_SYSTEM_PROMPT_IMPLEMENTATION.format(
        language=request.language)
      instruction = DEFAULT_INSTRUCTION_IMPLEMENTATION.format(
        instruction=request.instruction)
      response = self.llm_chain.run(system=system, instruction=instruction)
      blocks = self.extract_code_blocks(response)
      if len(blocks) > 0:
        db.insert_implementation(request.language, request.instruction,
                                 response)
        implementation = [block[1] for block in blocks]
        return Implementation(implementations=implementation)
      return Implementation(implementations=[])
    else:
      return Implementation(implementations=[])

  def GenerateTest(self, request: Code, context) -> Implementation:
    if self.instruct:
      system = DEFAULT_SYSTEM_PROMPT_TEST.format(language=request.language)
      instruction = DEFAULT_INSTRUCTION_TEST.format(
        language=request.language, code=request.code)
      response = self.llm_chain.run(system=system, instruction=instruction)
      blocks = self.extract_code_blocks(response)
      if len(blocks) > 0:
        db.insert_test_case(request.language, request.code, response)
        implementation = [block[1] for block in blocks]
        return Implementation(implementations=implementation)
      return Implementation(implementations=[])
    else:
      return Implementation(implementations=[])

  def GenerateExample(self, request: Code, context) -> Example:
    if self.instruct:
      system = DEFAULT_SYSTEM_PROMPT_EXAMPLE.format(language=request.language)
      instruction = DEFAULT_INSTRUCTION_EXAMPLE.format(
        language=request.language, code=request.code)
      response = self.llm_chain.run(system=system, instruction=instruction)
      blocks = self.extract_code_blocks(response)
      if len(blocks) > 0:
        db.insert_example(request.language, request.code, response)
        examples = [block[1] for block in blocks]
        return Example(examples=examples)
      return Example(examples=[])
    else:
      return Example(examples=[])


def is_port_in_use(port: int) -> bool:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    return s.connect_ex(('localhost', port)) == 0


def main():
  if is_port_in_use(args.port):
    logger.info('grpc server already started')
    return

  if not os.path.exists(args.model):
    logger.warn(f'model {args.model} not found')
    return

  logger.info('starting server...')
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  logger.info('startin servicer')
  servicer = MLServicer()
  ml_pb2_grpc.add_MLServicer_to_server(servicer, server)
  server.add_insecure_port(f'[::]:{args.port}')
  server.start()
  server.wait_for_termination()


if __name__ == '__main__':
  main()
