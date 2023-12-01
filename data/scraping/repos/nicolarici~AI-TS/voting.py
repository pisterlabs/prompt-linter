import time
import openai
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sorter import set_seed, predict_operator, get_ticket_prompt


######## CONSTANTS ########

OPENAI_KEY_FILE = "openai_key.txt"
TESTSET = "datasets/testset_1.json"

SEED = 224964

MODEL = 'gpt-4'

SYSTEM_PROMPTS = {
    "Human": "",
    "Categories": "",
    "Area": "",
    "Summaries": ""
}

if __name__ == '__main__':

    set_seed(SEED)

    # Initialize the OpenAI API from openai_key.txt.

    with open(OPENAI_KEY_FILE, "r") as f:
        openai.api_key = f.read()

    # Load the dataset
    df = pd.read_json(TESTSET).sample(n=6)

    # Predict the operator.                
    predictions = []

    for i, row in tqdm(df.iterrows(), total=len(df)):

        ticket_prompt = get_ticket_prompt(row.sCategoria, row.sOggetto, row.tTicket)

        preds = {}
        for type_prompt, sys_prompt in SYSTEM_PROMPTS.items():
            try:
                preds[type_prompt] = predict_operator(MODEL, sys_prompt, ticket_prompt)

            except openai.error.RateLimitError:
                print("Rate limit reached. Waiting 1 minute...")
                time.sleep(60)
                print("Continuing...")
                preds[type_prompt] = predict_operator(MODEL, sys_prompt, ticket_prompt)
                
            except openai.error.ServiceUnavailableError:
                print("The server is overloaded or not ready yet.")
                print("Waiting 30 seconds...")
                time.sleep(30)
                print("Continuing...")
                preds[type_prompt] = predict_operator(MODEL, sys_prompt, ticket_prompt)
            
            except Exception:
                print("General error")
                print("Waiting 30 seconds...")
                time.sleep(30)
                print("Continuing...")
                pred = predict_operator(MODEL, sys_prompt, ticket_prompt)
   

        # If tie, random the prediction.
        preds_values = list(preds.values())
        random.shuffle(preds_values)
        predictions.append(max(set(preds_values), key=preds_values.count))

        time.sleep(3)

    # Results.
    print(classification_report(df.sOperatore, predictions, zero_division=0))
