from langchain import LLMChain
import re
import time
import openai

class QABot():
  """
  Function to create the bot
  """
  def __init__(self, llm, retriever, prompt):
    self.llm = llm
    self.retriever = retriever
    self.prompt = prompt
    self.qa_chain = LLMChain(llm = self.llm, prompt=prompt)
    self.abbreviations = { # known abbreviations we want to replace
      "LSEG": "London Stock Exchange Group"
      } 


  def _is_good_answer(self, answer):

    ''' check if answer is a valid '''

    result = True # default response

    badanswer_phrases = [ # phrases that indicate model produced non-answer
      "no information", "no context", "don't know", "no clear answer", "sorry","not mentioned",
      "no answer", "no mention", "reminder", "context does not provide", "no helpful answer", "not specified","not know the answer",
      "given context", "no helpful", "no relevant", "no question", "not clear","not explicitly mentioned",
      "don't have enough information", " does not have the relevant information", "does not seem to be directly related"
      ]
    
    if answer is None: # bad answer if answer is none
      results = False
    else: # bad answer if contains badanswer phrase
      for phrase in badanswer_phrases:
        if phrase in answer.lower():
          result = False
          break
    
    return result


  def _get_answer(self, context, question, timeout_sec=60):

    '''' get answer from llm with timeout handling '''

    # default result
    result = None

    # define end time
    end_time = time.time() + timeout_sec

    # try timeout
    while time.time() < end_time:

      # attempt to get a response
      try: 
        result =  self.qa_chain.generate([{'context': context, 'question': question}])
        break # if successful response, stop looping

      # if rate limit error...
      except openai.error.RateLimitError as rate_limit_error:
        if time.time() < end_time: # if time permits, sleep
          time.sleep(2)
          continue
        else: # otherwise, raiser the exception
          raise rate_limit_error

      # if other error, raise it
      except Exception as e:
        print(f'LLM QA Chain encountered unexpected error: {e}')
        raise e

    return result


  def get_answer(self, question):
    ''' get answer to provided question '''

    # default result
    result = {'answer':None, 'source':None, 'output_metadata':None}

    # remove common abbreviations from question
    for abbreviation, full_text in self.abbreviations.items():
      pattern = re.compile(fr'\b({abbreviation}|{abbreviation.lower()})\b', re.IGNORECASE)
      question = pattern.sub(f"{abbreviation} ({full_text})", question)

    # get relevant documents
    docs = self.retriever.get_relevant_documents(question)

    # for each doc ...
    for doc in docs:

      # get key elements for doc
      text = doc.page_content
      source = doc.metadata['source']

      # get an answer from llm
      output = self._get_answer(text, question)
 
      # get output from results
      generation = output.generations[0][0]
      answer = generation.text
      output_metadata = output.llm_output

      # assemble results if not no_answer
      if self._is_good_answer(answer):
        result['answer'] = answer
        result['source'] = source
        result['output_metadata'] = output_metadata
        result['vector_doc'] = text
        break # stop looping if good answer
      
    return result