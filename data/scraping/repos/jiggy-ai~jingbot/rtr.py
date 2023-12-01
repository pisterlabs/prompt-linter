from jiggypedia import wikipedia_search as search
from langchain.llms import OpenAI
import openai  # for retry on error
#from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain
#from langchain.evaluation.qa import QAEvalChain
from typing import List
from concurrent.futures import ThreadPoolExecutor
from time import sleep, strftime
from loguru import logger

from dynamic_prompt import dynamic_prompt

MODEL_NAME="text-davinci-003"
K = 20
MAX_TOTAL_TOKENS = 3800
QA_TEMPERATURE = 0.125
QA_RESPONSE_TOKENS = 256
QA_RANDOMIZE = False



logger.info(f"K:  {K}")
logger.info(f"MAX_TOTAL_TOKENS:  {MAX_TOTAL_TOKENS}")
logger.info(f"QA_TEMPERATURE:  {QA_TEMPERATURE}")
logger.info(f"QA_RESPONSE_TOKENS:  {QA_RESPONSE_TOKENS}")
logger.info(f"MODEL_NAME:  {MODEL_NAME}")


# open ai offen returns rate limit error, so we need to retry
RETRY_COUNT = 10
def retry_llm(llm, prompt_text):
    for i in range(RETRY_COUNT):
        try:
            return llm(prompt_text)
        except (openai.error.RateLimitError, openai.error.ServiceUnavailableError):
            print("openai error, retrying", i)
            if i == RETRY_COUNT - 1:
                raise
            sleep(i**1.3)
        except Exception as e:
            print("failed to run llm", e)
        
# qa llm used to perform the task of answering the question based on given context
qa_llm = OpenAI(model_name=MODEL_NAME, temperature=QA_TEMPERATURE, max_tokens=400)

@dynamic_prompt(llm=qa_llm, response_tokens=QA_RESPONSE_TOKENS)
def qa_prompt(context : List[str],  question : str):
    prompt  = "Use the following Context to respond to the Question. "
    prompt += "Provide any additional relevant or interesting details from the Context. "
    prompt += "If the Context does not contain the required information for answering the Question "
    prompt += "then answer 'Not enough information'.\n"
    prompt += "If the Question is ambiguous or could have multiple answers, "
    prompt += "respond with a question that would help clarify the ambiguity.\n"
    prompt += "Context:\n"
    prompt += "\n".join(context)  
    prompt +=  f"\nQuestion: {question}\n"
    prompt += "Response: "
    return prompt

def ask(question, max_total_tokens=MAX_TOTAL_TOKENS, k=K):
    try:
        results = search(question, k=k, max_total_tokens=max_total_tokens)
    except Exception as ex:
        logger.exception("search error", ex)
        try:
            results = search(question, k=k, max_total_tokens=max_total_tokens)
        except Exception as ex:
            logger.exception("search error again", ex)
            return "Not enough information"
    total_tokens = sum(r.token_count for r in results)
    logger.info(f'total tokens {total_tokens} {len(results)}')
    context = [r.text for r in results]
    prompt_text = qa_prompt(context=context, question=question)
    logger.debug(prompt_text)
    ret = retry_llm(qa_llm, prompt_text)
    logger.debug(ret)
    return ret

from chatstack import SystemMessage, UserMessage, ContextMessage, completion


def askchat(question, max_total_tokens=MAX_TOTAL_TOKENS, k=K):
    try:
        results = search(question, k=k, max_total_tokens=max_total_tokens)
    except Exception as ex:
        logger.exception("search error", ex)
        try:
            results = search(question, k=k, max_total_tokens=max_total_tokens)
        except Exception as ex:
            logger.exception("search error again", ex)
            return "Not enough information"
    total_tokens = sum(r.token_count for r in results)
    logger.info(f'total tokens {total_tokens} {len(results)}')

    prompt  = "Use the following system context to respond to the User. "
    prompt += "Provide any additional relevant or interesting details from the system context. "
    prompt += "If the system context does not contain the required information for answering the user question "
    prompt += "then answer 'Not enough information'.\n"
    prompt += "If the user question is ambiguous or could have multiple answers, "
    prompt += "respond with a question that would help clarify the ambiguity."    

    messages  = [SystemMessage(text=prompt)]
    messages += [ContextMessage(text=r.text) for r in results]
    messages += [UserMessage(text=question)]

    return completion(messages, temperature=QA_TEMPERATURE)