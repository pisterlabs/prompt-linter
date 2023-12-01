import pandas as pd
import os
import argparse
import ast
import time
from typing import List
import google.auth
from langchain.chat_models import ChatVertexAI, ChatOpenAI
from lora_confirm import filters
from lora_confirm import question_generator as qgen
from reference_files import seeds
from reference_files import schema

# Instantiate the parser
parser = argparse.ArgumentParser(description=\
    "question generation script" \
    )
# Required arguments

parser.add_argument('--save_path', type=str, required=True,
                    help='Save path (directory) for the generated questions.')

parser.add_argument('--n', type=int, required=True,
                    help='Number of questions to be generated')

parser.add_argument('--llm', type=str, required=True,
                    help='LLM to generate questions. ONLY "gpt" and "palm" are implemented for this script.')

parser.add_argument('--google_credential_path', type=str, required=False,
                    default="gcp-service-account.json",
                    help='Path to the gcp-service-account.json. It is required if you use the palm model')

parser.add_argument('--openai_api_key', type=str, required=False,
                    default='None',
                    help='OpenAI API KEY. It is required if you use the model provided by OpenAI.')

parser.add_argument('--company_name', type=str, required=False,
                    default='None',
                    help='Company names to generate questions. Input with semicolon separated')

parser.add_argument('--fields', type=str, nargs='+',
                    help='Fields to generate questions. ONLY "all", "external" and "stock" are available now. Select one or more from them. Input with space separated')


def main(**kwargs):
    """Generate questions on selected fields and companies.
    
    Note: To use palm, you should have a valid service account and project. 
        You should have a valid credential file as in the "google_credential_path".
        Ref: https://cloud.google.com/vertex-ai/docs/start/cloud-environment
    """
    companies = kwargs['company_name']
    companies = companies.split(';')
    companies = [x.strip() for x in companies]
    n = kwargs['n']
    save_path = kwargs['save_path'] if kwargs['save_path'].endswith('/') else kwargs['save_path'] + '/'
    for field in kwargs['fields']:
        assert field in ['all', 'external', 'stock'], f'Invalid field {field}.ONLY "all", "external" and "stock" are available now. Select one or more from them.'

    assert kwargs['llm'] in ['gpt', 'palm'], f'Invalid llm {kwargs["llm"]}. ONLY "gpt" and "palm" are available now. Select one or more from them.'
    assert os.path.exists(save_path), "Save path does not exist"
    
    if kwargs['llm'] == 'palm':
        assert os.path.exists(kwargs['google_credential_path']), f"Invalid google credential path {kwargs['google_credential_path']}"
        credentials, project_id = google.auth.load_credentials_from_file(kwargs['google_credential_path'])
    elif kwargs['llm'] == 'gpt':
        assert kwargs['openai_api_key'] != 'None', "OpenAI API KEY is required if you use the model provided by OpenAI."
        os.environ['OPENAI_API_KEY'] = kwargs['openai_api_key']
    # Self-instruct used 0.7 temperature    
    llm = ChatOpenAI(temperature=0.7) if kwargs['llm'] == 'gpt' else ChatVertexAI(temperature=0.7)
    
    def _parse_expected_fields(x):
        """Auxiliary function to remove descriptions from expected fields and return a list of fields"""
        # Check if x is a string and looks like a stringified list
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            # Parse the stringified list
            x = ast.literal_eval(x)
        # Check if x is now a list (either it was originally, or it's been parsed from a string)
        if isinstance(x, list):
            # Process the list
            return [i.split(' - ')[0]for i in x]
        else:
            # If x is neither a list nor a stringified list, return None or handle the error as you see fit
            return None

    def gen_questions(
        llm,
        N,
        companies,
        fields,
        max_tries,
        batch_size,
        filename,
        seeds: List[str] = seeds.ALL_SEEDS,
    ): 
        tries = 0
        raw_file = f"{save_path}raw_{filename}" # Keep as checkpoints
        filtered_file = f"{save_path}filtered_{filename}"
        questions = pd.read_csv(raw_file) if os.path.exists(raw_file) else []
        filename = f"{save_path}{filename}"
        
        while len(questions) < N and tries < max_tries:
            # Generate questions
            while len(questions) < N * 1.3: # Buffer to account for filtering
                print("Generating questions..")
                try:
                    new_qs = qgen.seeded(llm=llm, 
                                        fields=fields, 
                                        subjects=companies,
                                        prompt_samples=seeds, 
                                        n=batch_size)
                    questions = pd.concat([questions, new_qs]) if len(questions) > 0 else new_qs
                    questions.to_csv(raw_file, index=False)
                    print(f"{len(questions)}/{N} questions generated")
                except Exception as e:
                    print(e)
                    time.sleep(4)
                
            # Filter (improve diversity of questions)
            print("Filtering questions..")
            questions = filters.rouge_l(questions, 
                                        columns=['question'], threshold=0.5) # self-instruct used .7, but our questions are shorter
            questions.reset_index(drop=True, inplace=True) #  Just to be pretty
            questions.to_csv(filtered_file, index=False)
            print(f"{len(questions)} questions after filtering")
            tries += 1
        questions = questions.head(N)
        print("Done!") if tries < max_tries else print("Max tries hit!")
        print(f"Generated {len(questions)} questions")
        
        # Clean up
        questions['expected_fields_with_descriptions'] = questions['expected_fields'] # This is a string
        questions['expected_fields'] = questions['expected_fields'].apply(_parse_expected_fields) # This is a list
        questions.to_csv(filename, index=False)
        os.remove(raw_file)
        os.remove(filtered_file)
    
    
    for field in kwargs['fields']:
        if field == 'stock':
            temp_fields = schema.stock_data
            temp_seeds = seeds.ALL_SEEDS
        elif field == 'external':
            temp_fields = schema.external_data_source
            temp_seeds = seeds.EXTERNAL_DATA
        elif field == 'all':
            temp_fields = schema.all_fields
            temp_seeds = seeds.ALL_SEEDS
        
        gen_questions(llm=llm,
                      N=n,
                      companies=companies,
                      fields=temp_fields,
                      max_tries=5,
                      batch_size=50,
                      seeds=temp_seeds,
                      filename=f"ConFIRM_QAset_{field}_fields_questions_{n}n_{kwargs['llm']}.csv")


if __name__=="__main__":
    args = parser.parse_args()
    main(**vars(args))
