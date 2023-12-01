import time
from dotenv import load_dotenv
import os
import re
from typing import Dict, List
import tiktoken as tiktoken
import importlib
import openai

# Load default environment variables (.env)
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")).lower()


openai.api_key = OPENAI_API_KEY
def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def limit_tokens_from_string(string: str, model: str, limit: int) -> str:
    """Limits the string to a number of tokens (estimated)."""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model('gpt2')  # Fallback for others.

    encoded = encoding.encode(string)

    return encoding.decode(encoded[:limit])


def try_pinecone():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    if PINECONE_API_KEY and can_import("extensions.pinecone_storage"):
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
        assert (
            PINECONE_ENVIRONMENT
        ), "\033[91m\033[1m" + "PINECONE_ENVIRONMENT environment variable is missing from .env" + "\033[0m\033[0m"
        from extensions.pinecone_storage import PineconeResultsStorage
        print("\nUsing results storage: " + "\033[93m\033[1m" + "Pinecone" + "\033[0m\033[0m")
        return PineconeResultsStorage(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, LLM_MODEL)
    return None





def context_agent(query: str, top_results_num: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        top_results_num (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    results = try_pinecone.query(query=query, top_results_num=top_results_num)
    # print("****RESULTS****")
    # print(results)
    return results

def openai_call(
    prompt,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
    messages: List[Dict[str, str]] = [],
):
    while True:
        try:
           
            if not model.lower().startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:

                trimmed_prompt = limit_tokens_from_string(prompt, model, 4000 - max_tokens)

                # Use chat completion API
                
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break

def essay_intro_agent(
        topic: str,
        message_list
):
    prompt = f"""You are an helpfull ai agent , your task is to write Introduction for a Essay Topic: {topic} \n

                1. Begin with an attention-grabbing hook or statement.
                2. Provide background information or context related to the topic.
                3. Present the thesis statement, which is the main argument or point of the essay.

                """

    # print(f'\n*****ESSAY INTRODUCTION AGENT PROMPT****\n{prompt}\n')
    message_list.append({"role": "system", "content": prompt})
   
    response = openai_call(prompt, max_tokens=2000)

    message_list.append({"role": "user", "content": response})
    # print(f'\n****ESSAY INTRODUCTION AGENT RESPONSE****\n{response}\n')
    
    return message_list


def essay_body_agent(
        essay_introduction: str,
        message_list
):
    prompt = f"""You are an helpfull ai agent , your task is to write Body Paragraphs for a Essay Topic Intro: {essay_introduction} \n

                1. Each paragraph should start with a clear topic sentence that relates to the thesis.
                2. Provide evidence, examples, or arguments to support the topic sentence.
                3. Use transitions to connect paragraphs and ideas.
                4. Each paragraph should focus on a single point or idea.
                5. You can have as many body paragraphs as needed, but a common structure is three to five paragraphs for shorter essays.

                """

    # print(f'\n*****ESSAY BODY AGENT PROMPT****\n{prompt}\n')
    
   
    response = openai_call(prompt, max_tokens=2000)

   
    # print(f'\n****ESSAY BODY AGENT RESPONSE****\n{response}\n')
    
    return response





def main():
   message_list  = []
   intro_list = essay_intro_agent("Artificial Intelligence",message_list)

   essay_introduction = intro_list[len(intro_list)-1]["content"]
   


   body  = essay_body_agent(essay_introduction,list)

#    essay_body = body[len(body)-1]["content"]

#    print(f'\n****ESSAY****\n{essay_introduction}\n{essay_body}')






   
 





if __name__ == "__main__":
 main()