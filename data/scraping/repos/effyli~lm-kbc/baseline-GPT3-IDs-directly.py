import json
import openai
import ast
from file_io import *
from evaluate import *
import time
import argparse


def GPT3response(q):
    response = openai.Completion.create(
        # curie is factor of 10 cheaper than davinci, but correspondingly less performant
        model="text-davinci-003",
        #model = "text-curie-001",
        prompt=q,
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )    
    response = response.choices[0].text
    if response[0] == " ":
        response = response[1:]
    try:    
        response = ast.literal_eval(response)
    except:
        response = []
    return response
        

def run(args):
    openai.api_key = args.oaikey
    
    prefix = '''State of Palestine, country-borders-country, ["Q801"]
    Paraguay, country-borders-country, ["Q155", "Q414", "Q750"]
    Lithuania, country-borders-country, ["Q34", "Q36", "Q159", "Q184", "Q211"]
    ''' 

    print('Starting probing GPT-3 ................')

    train_df = read_lm_kbc_jsonl_to_df(Path(args.input))
    
    print (train_df)

    results = []
    for idx, row in train_df.iterrows():
        prompt = prefix + row["SubjectEntity"] + ", " + row["Relation"] + ", "
        print("Prompt is \"{}\"".format(prompt))
        result = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntitiesID": GPT3response(prompt),  ## naming with IDs required for current evaluation script 
        }
        results.append(result)

    save_df_to_jsonl(Path(args.output), results)

    print('Finished probing GPT_3 ................')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Model with Question and Fill-Mask Prompts")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input (subjects) file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Predictions (output) file")
    parser.add_argument("-k", "--oaikey", type=str, required=True, help="OpenAI API key")

    args = parser.parse_args()

    run(args)