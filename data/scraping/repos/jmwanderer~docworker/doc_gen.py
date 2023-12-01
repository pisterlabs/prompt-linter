"""
Utilitiy to work with information from DOCX files and the OpenAI
LLM. This utility supports:
- reading a docx file and intelligently separating into sections
- supporting completions for each section
- completion of the completion results
- saving state
"""
from . import section_util
from . import document
import logging
import datetime
import math
import time
import tiktoken
import openai


# Global Flag to mock the AI call
FAKE_AI_COMPLETION=False
# Global value for sleep duration in a mock AI call
FAKE_AI_SLEEP=5

# Global value for base timeout in seconds
AI_BASE_TIMEOUT=60


class ResponseRecord:
  def __init__(self,
               text,
               prompt_tokens,
               completion_tokens,
               truncated,
               timeout_value):
    self.text = text
    self.prompt_tokens = prompt_tokens
    self.completion_tokens = completion_tokens
    self.truncated = truncated
    self.timeout_value = timeout_value
    
               
def run_completion(prompt, text, max_tokens, timeout_value, status_cb=None):
  """
  Run an AI completion with the given prompt and text.

  max_tokens may limit the return size or be set to -1
  if status_cb set, called with updates
  """

  prompt = build_prompt(prompt)
  tokenizer = tiktoken.encoding_for_model(section_util.AI_MODEL)
  text = "\"\"" + text + "\"\"\""

  done = False
  max_try = 5
  count = 0
  request_timeout = 0 
  wait = 5
  completion = None
  truncated = False
  completion_token_count = 0
  completion_cost = 0
  prompt_tokens = len(tokenizer.encode(prompt))
  text_tokens = len(tokenizer.encode(text))

  # Enusre the total request is less than the max
  # TODO: more accurate calc of size, don't use factor of 50
  limit_tokens =  (section_util.AI_MODEL_SIZE -
                  prompt_tokens - text_tokens - 50)
  if max_tokens == -1 or max_tokens > limit_tokens:
    max_tokens = limit_tokens

  base_timeout = timeout_value
  if base_timeout == 0:
    base_timeout = AI_BASE_TIMEOUT    

  logging.info("prompt tokens: %d, text tokens: %d, max_tokens: %d, timeout: %d" %
               (prompt_tokens, text_tokens, max_tokens, base_timeout))
  
  logging.info("Running completion: %s", prompt)  
  if FAKE_AI_COMPLETION:
    time.sleep(FAKE_AI_SLEEP)
    return ResponseRecord("Dummy completion, this is filler text.\n" * 20,
                          prompt_tokens = 150,
                          completion_tokens = 50,
                          truncated=False,
                          timeout_value = 0)

  # Loop with exponential retries
  while not done:
    try:
      start_time = datetime.datetime.now()
      request_timeout = base_timeout + (15 * count)
      response = openai.ChatCompletion.create(
        model=section_util.AI_MODEL,
        max_tokens = max_tokens,
        temperature = 0.1,
        messages=[ {"role": "system", "content": prompt},
                   {"role": "user", "content": text }],
        request_timeout=request_timeout)
      
      completion = response['choices'][0]['message']['content']
      completion_tokens = response['usage']['completion_tokens']
      prompt_tokens = response['usage']['prompt_tokens']            
      done = True
      if response['choices'][0]['finish_reason'] == "length":
        truncated = True
      end_time = datetime.datetime.now()

    except Exception as err:
      end_time = datetime.datetime.now()
      logging.error(str(err))      
      if status_cb is not None:
        status_cb(str(err))

    logging.info("completion required %d seconds" %
                 (end_time - start_time).total_seconds())
    count += 1
    if count >= max_try and not done:
      completion = ""
      prompt_tokens = 0
      completion_tokens = 0
      truncated = False
      done = True
        
    if not done:
      wait_time = wait * math.pow(2, count - 1)
      time.sleep(wait_time)

  return ResponseRecord(completion, prompt_tokens,
                        completion_tokens, truncated, request_timeout)


class RunState:
  """
  State of running completion task.
  
  Tracks all state needed to run a completion task.
  """
  def __init__(self):
    # Original list of item ids for processing
    self.source_items = []

    # List of items ids to be processed in this round
    self.to_run = []

    # List of item ids that have been produced in this round
    self.current_results = []

    # List of item ids that have been produced
    self.completed_run = []

    # Prompt for AI completion
    self.prompt = ""

    # ID of the final resulting item
    self.result_id = 0

    # Run ID as recorded in the Document
    self.run_id = 0

    # True if this is a tranform instead of consolidate proces.
    self.op_type = document.OP_TYPE_CONSOLIDATE

    # Last timeout theshold used where the call suceeded
    self.timeout_value = 0

    
  def start_run(self, prompt, item_ids, run_id, op_type):
    """
    Setup to begin the run process.
    """
    self.run_id = run_id
    self.prompt = prompt
    self.to_run = item_ids.copy()
    self.source_items = item_ids.copy()
    self.op_type = op_type

  def next_item(self):
    """
    Return the next item to be processed, None if empty
    """
    if len(self.to_run) == 0:
      return None
    return self.to_run[0]

  def pop_item(self):
    """
    Remove the next item to be run and return the ID
    """
    id = self.to_run[0]
    self.to_run.pop(0)
    return id
  
  def skip_remaining_gen(self):
    """
    Detect case where one intermediate result is on the to_run list.
    Does not make sense to re-generate a single result for this process.
    """
    # Check if there is one item and it is not a source_id
    return (len(self.to_run) == 1 and
            not self.to_run[0] in self.source_items)
    
  def note_step_completed(self, result_id):
    """
    Records the result of a completion
    """
    self.completed_run.append(result_id)
    self.current_results.append(result_id)

  def next_result_set(self):
    """
    Switch to next set to process and return 
    True if there are more items to process.
    """
    if len(self.current_results) == 1:
      # End condition, one result produced.
      self.result_id = self.current_results[0]
      self.current_results = []
      return False

    if len(self.current_results) > 1:
      # More results to process 
      self.to_run = self.current_results
      self.current_results = []
      return True

    return False

  def is_last_completion(self):
    """
    Return true if the completion to be run will be the last.
    """
    return len(self.to_run) == 0 and len(self.current_results) == 0
    
  def in_additions(self, item_id):
    return (item_id != self.result_id and
            item_id in self.completed_run)

  def get_source_items(self):
    return self.source_items


def build_prompt(prompt):
  """
  Extend the given prompt to something that is effective for 
  GPT completions.
  """
  return  "You will be provided with text delimited by triple quotes. Using all of the text, " + prompt



def post_process_completion(response_record):
  """
  Make any fixups needed to the text after a completion run.
  """
  # Detect truncated output and truncate to the previous period or CR
  if response_record.truncated:
    last_cr = response_record.text.rfind('\n')
    if last_cr == -1:
      return
    logging.info("truncated response. from %d to %d" %
                 (len(response_record.text), last_cr + 1))
    response_record.text = response_record.text[0:last_cr + 1]

    
def start_docgen(file_path, doc, prompt, src_run_id=None,
                 op_type=document.OP_TYPE_CONSOLIDATE):
  """
  Setup state for a docgen run.
  Returns a run_state to use in further calls.
  """
  run_state = RunState()
    
  run_id = doc.mark_start_run(prompt, src_run_id, op_type)
  # Run on all doc segments by default.
  item_ids = []
  for item in doc.get_ordered_items(run_id):
    item_ids.append(item.id())
        
  run_state.start_run(prompt, item_ids, run_id, op_type)
  document.save_document(file_path, doc)
  return run_state


def run_input_tokens(doc, run_state):
  """
  Return the total number of tokens in the input
  """
  count = 0
  for item_id in run_state.to_run:
    item = doc.get_item_by_id(run_state.run_id, item_id)
    if item is not None:
      count += item.token_count()
  return count


def run_next_docgen(file_path, doc, run_state):
  """
  Called by run_all_docgen to make one completion.
  
  Combines as many items on the todo list as possible.
  """
  tokenizer = section_util.get_tokenizer()
  done = False
  item_id_list = []
  text_list = []
  token_count = 0
  status_message = ""
  
  # Pull items until max size would be exceeded
  while not done:
    item_id = run_state.next_item()
    if item_id is None:
      done = True
      continue

    item = doc.get_item_by_id(run_state.run_id, item_id)
    if item is None:
      run_state.pop_item()
      continue
    
    count = len(tokenizer.encode(item.text()))
    if (token_count != 0 and
        count + token_count > section_util.TEXT_EMBEDDING_CHUNK_SIZE):
      logging.debug("max would be hit, count = %d, token_count = %d" %
                    (count, token_count))
      done = True
      continue

    # Included the next item, remove from to do list
    run_state.pop_item()

    item_id_list.append(item.id())
    text_list.append(item.text())
    token_count += count
    logging.debug("add id %d to source list. count = %d, total = %d" %
                  (item.id(), count, token_count))
    # Build status message to include this item.
    if len(status_message) > 0:
      status_message += ", "
    status_message += item.name()

  if len(item_id_list) == 0:
    logging.error("No content for docgen")
    return

  # Setup to run a completion
  prompt = run_state.prompt
  prompt_id = doc.prompts.get_prompt_id(prompt)
                                        
  logging.info("run completion with %d items" % len(item_id_list))

  # Update status with last item
  doc.set_status_message("%s on %s" %
                         (prompt, status_message), run_state.run_id)
  document.save_document(file_path, doc)
  
  err_message = ''
  def status_cb(message):
    err_message = str(message)
    doc.set_status_message(str(message), run_state.run_id)
    document.save_document(file_path, doc)

  # Ensure response is less than 1/2 the size of a request
  # to make progress on consolidation. Except on the last completion.
  max_tokens = int(section_util.TEXT_EMBEDDING_CHUNK_SIZE / 2) - 1
  if (run_state.is_last_completion() or
      run_state.op_type == document.OP_TYPE_TRANSFORM):
    max_tokens = -1

  response_record = run_completion(prompt, '\n'.join(text_list),
                                   max_tokens,
                                   run_state.timeout_value,
                                   status_cb)

  run_state.timeout_value = response_record.timeout_value
  post_process_completion(response_record)
  
  if response_record.text is None:
    text = err_message
  else:
    text = response_record.text
    
  completion = doc.add_new_completion(
    item_id_list,
    text,
    response_record.completion_tokens,
    response_record.prompt_tokens + response_record.completion_tokens)
  run_state.note_step_completed(completion.id())


def combine_results(doc, run_state):
  # combine the results of all items on the complete run list
  # and make it the final result
  item_id_list = []
  text_list = []
  while run_state.next_item():
    id = run_state.pop_item()
    item_id_list.append(id)
    completion = doc.get_item_by_id(run_state.run_id, id)
    text_list.append(completion.text() + '\n')

  prompt = run_state.prompt
  prompt_id = doc.prompts.get_prompt_id(prompt)
  text = '\n'.join(text_list)
  tokenizer = tiktoken.encoding_for_model(section_util.AI_MODEL)    
  text_tokens = len(tokenizer.encode(text))
    
  completion = doc.add_new_completion(
    item_id_list,
    ''.join(text_list),
    text_tokens, 0)
  run_state.note_step_completed(completion.id())

  
def run_all_docgen(file_path, doc, run_state):
  #
  # Run a Doc Gen process that has been initialized with
  # start doc gen. Return the id of the results.
  #
  # Save the file as progress is made so the status can be
  # read.
  #
  done = False
  result_id = None
  doc.set_status_message("Running...")
  while not done:
    logging.debug("loop to run a set of docgen ops")      

    # loop to consume all run items
    while run_state.next_item() is not None:
      if run_state.skip_remaining_gen():
        # Skip procesing an already processed item
        id = run_state.pop_item()
        logging.debug("skip unnecessary docgen: %d", id)
        # Add directly to results list for further processing
        # TODO: is this a bug with result_id here and id above?
        run_state.note_step_completed(result_id)
      else:
        logging.debug("loop for running docgen")
        run_next_docgen(file_path, doc, run_state)
        document.save_document(file_path, doc)

    # If this a tranform, we combine the results into a final
    # and we are done
    if run_state.op_type == document.OP_TYPE_TRANSFORM:
      if run_state.next_result_set():
        combine_results(doc, run_state)

    # Done with the to_run queue, check if we process the
    # set of generated results.
    if not run_state.next_result_set():
      logging.info("doc gen complete")      
      done = True

  # Complete - mark final result and save
  completion = doc.get_item_by_id(run_state.run_id, run_state.result_id)
  if completion is not None:
    # TODO: make these less redundent 
    doc.set_final_result(completion)
    completion.set_final_result()    
    result_id = completion.id()

  doc.set_status_message("")
  doc.mark_complete_run()
  document.save_document(file_path, doc)
  return result_id
  
      

  
    
