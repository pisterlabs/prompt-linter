


# == ACTIONS == 
# --- desc: base functions for one-shot actions such as summarization

# -- standard imports
from retry import retry 

from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory

# -- TODO wrap wheere this is used 
from langchain.chains import ConversationalRetrievalChain

# -- local imports
from ._api import (loadLLM,
                   generateDocs,)

from ._chains import (loadSummaryChain,
                      loadLLMchain,
                      generationChain,)


# ==================================================
# -- LLM Generation 
# ==================================================
@retry(tries=3, delay=3)
def generateTextLLM(prompt, inputs, model = '', max_runs = 5, temp = 0.4, output_parser = None):
  # inputs: PromptTemplate, inputs dict
  # outputs: generated text 

  # -- initialize chain
  chain = generationChain(prompt, list(inputs.keys()), model = model, temp = temp, output_parser = output_parser)

  # -- generate text
  result = chain.generate([inputs])
  text = result.generations[0][0].text

  # -- check for continuation
  if type(result.generations[0][0].generation_info) != dict:
    return text
  
  finish_reason = result.generations[0][0].generation_info['finish_reason']
  run_count = 1

  if finish_reason == 'length':
    # -- re-initialize prompt + chain 
    prompt_prefix = 'Provided is an unfinished output. Please continue generating the output. '
    prompt_suffix = '{previous_output}'
    prompt_continue = prompt_prefix + prompt + prompt_suffix

    inputs = {**inputs, 'previous_output': text}

    #chain = generationChain(prompt_continue, list(inputs.keys()), model = model, temp = temp, output_parser = output_parser)

    while finish_reason == 'length' and run_count <= max_runs:
      text_ = result.generations[0][0].text
      inputs['previous_output'] = text_

      chain = generationChain(prompt_continue, list(inputs.keys()), model = model, temp = temp, output_parser = output_parser)

      result = chain.generate([inputs])

      text += result.generations[0][0].text

      finish_reason = result.generations[0][0].generation_info['finish_reason']
      run_count += 1


  if finish_reason == 'length':
    print('Warning: section text generation did not finish due to length.')

  return text


# ==================================================
# -- Summarizers 
# ==================================================

@retry(ValueError, tries=3, delay=3)
def summarizeText(text, prompt= '', input_variables = ["text"], temp = 0.2, chuck_size=1500, overlap=0, chain_type="map_reduce"):
  
  # -- define llm 
  llm = loadLLM(temp=temp)

  # -- load text
  docs = generateDocs(text, chuck_size=chuck_size, overlap=overlap)
                      
  # -- load chain 
  #prompt = prompt if prompt != '' else "Summarize the following text: {text}"
  prompt = PromptTemplate(template=prompt, input_variables=input_variables) if prompt != None else prompt
  chain = loadSummaryChain(llm, chain_type=chain_type, combine_prompt=prompt)

  # -- run chain
  summary = chain({"input_documents": docs}, return_only_outputs=True)

  # -- unpack summary
  summary = summary['output_text'] if 'output_text' in summary.keys() else ''

  # -- error handle or retry if summary is empty
  if len(summary) == 0:
    raise ValueError('Generated summary is empty.')
  
  return summary

# ==================================================
# -- Vectorstore  
# ==================================================
def chatVectorstore(vectorstore, query, memory = [], return_memory = False, model = 'chat', temp = 0.4):

  # -- initialize memory 
  if type(memory) != ConversationBufferMemory and return_memory:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

  # -- define chain 
  llm = loadLLM(temp=temp, model=model)
  qa = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(),
    #condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k'),
    memory=memory
  )

  result = qa({"question": query})

  # -- update memory. dont know if i need to return it or not 
  memory.save_context({"input": query}, {"output": result['answer']})

  if return_memory:
    return result['answer'], memory

  return result['answer'] 


# ==================================================
# -- Tags   
# ==================================================