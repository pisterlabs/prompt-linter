import os
import time

import openai
from tqdm import tqdm

import helper as utl


def call_openai(model, data, limit=1, count=1):
    """
    Calls the OpenAI API (https://platform.openai.com/docs/introduction) to extract unique names
    of organizations in the passed-in data. It provides the option to use recursion to pass the organization
    names through the OpenAI API more than once to try to reduce the number of errors in the final output.

    A list of OpenAI models can be found here:
    https://platform.openai.com/docs/models/model-endpoint-compatibility

    Error handling code is based on API documentation:
    https://platform.openai.com/docs/guides/error-codes/python-library-error-types

    :param model: (str) OpenAI chat model to use for correction.
    :param data: (list) list of organization names to parse.
    :param limit: (int) maximum number of iterations.
    :param count: (int) counter that updates with each iteration and terminates recursion when it equals limit.
    :return: (list) list of companies, organizations, and institutions.
    """
    orgs = set()
    for chunk in tqdm(data):
        try:
            prompt = f"Remove job titles such as 'Professor' or 'Research Scientist' or 'Software Engineer' in the\
            following list. Only give me organization names, and separate each organization name with a comma: {chunk}"
            openai.api_key = os.getenv('OPENAI_API_KEY')
            response = (openai.ChatCompletion.create(
                model=model,
                messages=[
                    {'role': 'system',
                     'content': 'You are a helpful assistant that is extracting company names from text. You do not \
                     add or invent new information. The answer must be contained in the text you are given.'},
                    {'role': 'user', 'content': prompt}
                ]
            )['choices'][0]['message']['content'])
            for entity in response.split(', '):
                entity = entity.replace('.', '')
                orgs.add(entity)
            time.sleep(20)
        except openai.error.APIError as e:
            print(f"API error: {e}")
        except openai.error.APIConnectionError as e:
            print(f"API connection failed: {e}")
        except openai.error.RateLimitError as e:
            print(f"Exceeded rate limit: {e}")
        except openai.error.AuthenticationError as e:
            print(f"Credential error: {e}")
        except openai.error.InvalidRequestError as e:
            print(f"Exceeded token limit: {e}")
    if count >= limit:
        return list(orgs)
    else:
        count += 1
        chunked = chunk_data(list(orgs), 300)
        return call_openai(model='gpt-3.5-turbo', data=chunked, count=count, limit=limit)


def chunk_data(data, size):
    """
    Splits data into chunks less than or equal to the length specified in size.
    :param data: (list) data to split into chunks.
    :param size: desired size of chunks.
    :return: list of lists.
    """
    chunks = [data[i:i + size] for i in range(0, len(data), size)]
    return chunks


def main():
    """
    Entry point for program.
    :parameter: none.
    :return: none.
    """

    # Load data from cache
    auths_coauths = utl.read_json('cache.json').get('auths-coauths')

    # Get institutional affiliations data from cache
    affils = []
    for faculty in auths_coauths:
        try:
            for coauth in faculty.get('coauthors'):
                affils.append(coauth.get('affiliation'))
        except AttributeError:
            continue
        except TypeError:
            continue

    # Use OpenAI API to parse institutions
    chunked = chunk_data(affils, 300)
    entities = call_openai(model='gpt-3.5-turbo', data=chunked, limit=2)
    # utl.update_cache('cache.json', entities, key='institutions')


if __name__ == '__main__':
    main()
