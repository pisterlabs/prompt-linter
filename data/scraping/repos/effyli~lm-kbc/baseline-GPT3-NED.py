import json
import openai
import ast
from file_io import *
from evaluate import *
import time
import argparse
import requests

# This baseline uses GPT-3 to generate surface forms, and Wikidata's disambiguation API to produce entity identifiers

# Get an answer from the GPT-API
def GPT3response(q):
    response = openai.Completion.create(
        # curie is factor of 10 cheaper than davinci, but correspondingly less performant
        #model="text-davinci-003",
        model = "text-curie-001",
        prompt=q,
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )    
    response = response.choices[0].text
    response = response.splitlines()[0]
    if len(response)>0:
        if response[0] == " ":
            response = response[1:]
    print("Answer is \"" + response + "\"\n")
    try:    
        response = ast.literal_eval(response)
    except:
        response = []
    return response
        

def disambiguation_baseline(item):
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
        data = requests.get(url).json()
        # Return the first id (Could upgrade this in the future)
        return data['search'][0]['id']
    except:
        return item

def run(args):
    openai.api_key = args.oaikey
    
    prefix = '''Paraguay, country-borders-country, ["Bolivia", "Brazil", "Argentina"]
    Cologne, CityLocatedAtRiver, ["Rhine"]
    Hexadecane, CompoundHasParts, ["carbon", "hydrogen"]
    Antoine Griezmann, FootballerPlaysPosition, ["forward"]
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
            "ObjectEntitiesSurfaceForms": GPT3response(prompt), 
            "ObjectEntitiesID": []
        }
        # special treatment of numeric relations, do not execute disambiguation
        if result["Relation"]=="PersonHasNumberOfChildren" or result["Relation"]=="SeriesHasNumberOfEpisodes":
            result["ObjectEntitiesID"] = result["ObjectEntitiesSurfaceForms"]
        # normal relations: execute Wikidata's disambiguation
        else:
            for s in result['ObjectEntitiesSurfaceForms']:
                result["ObjectEntitiesID"].append(disambiguation_baseline(s))

        results.append(result)

    save_df_to_jsonl(Path(args.output), results)

    print('Finished probing GPT_3 ................')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Model with Question and Fill-Mask Prompts")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input (subjects) file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Predictions (output) file")
    parser.add_argument("-k", "--oaikey", type=str, required=True, help="OpenAI API key")
    #parser.add_argument("-f", "--few_shot", type=int, default=5, help="Number of few-shot examples (default: 5)")
    #parser.add_argument("--train_data", type=str, required=True, help="CSV file containing train data for few-shot examples (required)")

    args = parser.parse_args()

    run(args)