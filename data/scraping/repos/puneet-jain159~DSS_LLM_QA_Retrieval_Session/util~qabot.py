from langchain import LLMChain
import re
import time
import openai

class QABot():


  def __init__(self, llm, retriever, prompt ,club_chunks = 3):
    self.llm = llm
    self.retriever = retriever
    self.prompt = prompt
    self.qa_chain = LLMChain(llm = self.llm, prompt=prompt)

  def _is_good_answer(self, answer):

    ''' check if answer is a valid '''

    result = True # default response

    badanswer_phrases = [ # phrases that indicate model produced non-answer
      "no information", "no context", "don't know", "no clear answer", "sorry","not mentioned","do not know","i don't see any information","i cannot provide information",
      "no answer", "no mention","not mentioned","not mention", "context does not provide", "no helpful answer", "not specified","not know the answer", 
      "no helpful", "no relevant", "no question", "not clear","provide me with the actual context document",
      "i'm ready to assist","I can answer the following questions"
      "don't have enough information", " does not have the relevant information", "does not seem to be directly related","cannot determine"
      ]
    if answer is None: # bad answer if answer is none
      results = False
    else: # bad answer if contains badanswer phrase
      for phrase in badanswer_phrases:
        if phrase in answer.lower():
          result = False
          break
    if answer[-1] == "?":
      result = False
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

    retriever_addon = "Represent this sentence for searching relevant passages: \n"

    # get relevant documents
    docs = self.retriever.get_relevant_documents(retriever_addon + question)

    # for each doc ...

    for x in range(0,len(docs),3):
      text = ""
      print(x,x+3)
      for doc in docs[x:x+3]:
        text += "\nParagraph: \n" + doc.page_content
    # print(text)

      # get key elements for doc
      # text = doc.page_content
      source = doc.metadata['source']

      # get an answer from llm
      output = self._get_answer(text, question)

      # get output from results
      generation = output.generations[0][0]
      answer = generation.text
      print("answer:",answer)
      output_metadata = output.llm_output

      # assemble results if not no_answer
      if self._is_good_answer(answer):
        result['answer'] = answer
        result['source'] = source
        result['output_metadata'] = output_metadata
        result['vector_doc'] = text
        return result
      else:
        result['answer'] = "Could not fine answer please rephrase the question or provide more context?"
        result['source'] = "NA"
        result['output_metadata'] = "NA"
        result['vector_doc'] = "NA"
        # print("text:",text)
      
    return result