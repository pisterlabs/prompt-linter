import torch
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from queue import Empty
from multiprocessing import Queue
import logging
import re

logging.basicConfig(level=logging.INFO)


template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: 
{instruction}

Answer:"""

def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
  if input_ctxt:
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_ctxt}

### Response:"""
  else:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def run_alpaca_damn_you(exit_signal, input_queue, output_queue):

  tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
  model = LlamaForCausalLM.from_pretrained(
    "chainyo/alpaca-lora-7b",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
  )

  pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
#    max_length=1024,
    temperature=0.8,
    top_p=0.75,
    top_k=40,
    repetition_penalty=1.2,
    max_new_tokens=512
  )

  local_llm = HuggingFacePipeline(pipeline=pipe)
  new_prompt = PromptTemplate(template = template, input_variables = ["instruction"])
#  llm_chain = LLMChain(prompt=new_prompt, llm=local_llm)
  window_memory = ConversationBufferWindowMemory(k=6)
  conversation = ConversationChain(llm=local_llm, verbose=True, memory=window_memory)

  """
  generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
  )

  model.eval()
  if torch.__version__ >= "2":
    model = torch.compile(model)

  instruction = "You are a highly skilled professional interviewing for a job."
  input_ctxt = ""
  """

  while not exit_signal.is_set():
    try:
      prompt_in = input_queue.get(timeout=1)
      logging.info(f"Asking alpaca: {prompt_in}")
      """
      input_ctxt = f"{input_ctxt}INTERVIEWER: {prompt_in}\nCANDIDATE: \n"
      prompt = generate_prompt(instruction, input_ctxt)
      logging.info(f"alpaca prompt: {prompt}")
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids
      input_ids = input_ids.to(model.device)

      with torch.no_grad():
        outputs = model.generate(
          input_ids=input_ids,
          generation_config=generation_config,
          return_dict_in_generate=True,
          output_scores=True,
        )

      full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
      response = re.sub('.+Response:', '', full_response, flags=re.DOTALL)

      logging.info(f"Alpaca context: {input_ctxt}")
      logging.info(f"Alpaca response: {response}")
      input_ctxt = f"{input_ctxt}{response}\n"
      """
      full_response=conversation(prompt_in)
      logging.info(full_response)
      response=full_response.get('response').strip()
      if response != '':
        output_queue.put(response)
    except Empty:
      pass
