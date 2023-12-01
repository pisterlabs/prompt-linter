import argparse
import openai as oa

from python_template import logger
from multiprocessing.dummy import Pool

"""

    References
        - https://platform.openai.com/docs/api-reference/introduction?lang=python

"""

def send_prompts(prompt):

    try:

        response = oa.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=1000
                )
        return {
                "response": response,
                "outcome": "success"
        }
    
    # not tests
    except openai.error.RateLimitError as e:
        logger.error('rate-limit-error', e)
        return {
                "response": None,
                "outcome":"failed"
         }

    except Exception as e:
        logger.error('unknown error', e)
        return {
                "response": None,
                "outcome": "faied"
        }

if __name__ == "__main__":

    # parse arguments
    logger.info('start tests')

    # start
    oa.api_key = ""
    prompts = ["How are you?", "What is your name?", "What is the meaning of life?"]

    with Pool() as pool:
        results = pool.map(send_prompts, prompts)

        for res in results:
            # results.append([intitle, resp])
            # print(f'{prompt}: {resp}')
            logger.info(res)

    logger.info('tests complete') 

