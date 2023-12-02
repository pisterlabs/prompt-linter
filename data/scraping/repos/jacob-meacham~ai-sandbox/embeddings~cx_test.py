import argparse
import csv

import langchain.llms

from embeddings.cx_bot import PROMPT_TEMPLATE, get_embeddings
from embeddings.llm_with_embeddings import ModelWithEmbeddings
from embeddings.openai_utils import get_openai_api_key

MODEL = 'gpt-3.5-turbo'

def main(docs_dir, cache_dir, input_requests, output_responses):
    llm = langchain.llms.OpenAI(temperature=0.7, model_name=MODEL, openai_api_key=get_openai_api_key())
    embeddings_db = get_embeddings(docs_dir, cache_dir)

    model = ModelWithEmbeddings(llm, embeddings_db, PROMPT_TEMPLATE)

    tests = []
    with open(input_requests, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            match row['platform']:
                case 'MINDBODY':
                    context = {
                        'phone': '1-877-755-4279',
                        'customer_tier': row['tier'] if row['tier'] else 'Accelerate',
                        'customer_platform': 'MINDBODY',
                        'features': row['features'].split(',') if row['features'] else 'This customer uses the New Mindbody Experience'
                    }
                case 'Booker':
                    context = {
                        'phone': '1-866-966-9798',
                        'customer_tier': 'V1',
                        'customer_platform': 'Booker',
                        'features': []
                    }

            tests.append({
                'query': row['query'],
                'customer_platform': row['platform'],
                'context': context
            })

    responses = []
    for t in tests:
        response = model.submit_query(t['query'], t['context'], {'product': t['customer_platform']})
        print(response)
        responses.append(response)

    defaults = {
        'Was the Response Accurate': '',
        'Response Rating': '',
        'How would you improve this reponse': '',
        'Any Addtional Notes/Comments': '',
    }
    output_rows = [{
        'Question': t['query'],
        'Platform': t['customer_platform'],
        'Context': str(t['context']),
        'Response': response,
        **defaults
    } for t, response in zip(tests, responses)]

    with open(output_responses, 'w') as f:
        writer = csv.DictWriter(f, output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('docs_dir')
    parser.add_argument('input_requests')
    parser.add_argument('output_responses')
    parser.add_argument('--cache-dir', default='./.cx_cache')
    parser.add_argument('--force-cache-rebuild', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_options()
    main(options.docs_dir, options.cache_dir, options.input_requests, options.output_responses)