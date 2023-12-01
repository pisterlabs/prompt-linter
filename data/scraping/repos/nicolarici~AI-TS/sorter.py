import os
import time
import random
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report


######## FUCTIONS ########

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def predict_operator(model, system_prompt, ticket_prompt):

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ticket_prompt}
        ],
        temperature=0,
        top_p=0.01,
        max_tokens=3
    )

    return completion.choices[0].message.content



def get_ticket_prompt(ticket, categoria, oggetto):

    prompt  = f"Nuovo Ticket: \n"
    prompt += f"CATEGORIA: {categoria} \n"
    prompt += f"OGGETTO: {oggetto} \n"    
    prompt += f"DESCRIZIONE: {ticket} \n"

    return prompt


def get_few_shot_prompt(df, n):

    labels = df.sOperatore.unique().tolist()
    few_shot = ""

    for l in labels:
        tmp = df[df.sOperatore == l].sample(n, random_state=SEED)

        for _, row in tmp.iterrows():
            few_shot += get_ticket_prompt(row.sCategoria, row.sOggetto, row.tTicket)
            few_shot += f"{l} \n\n"

    return few_shot

def summarize_tickets(model, df_macro, num_ticket):

    summaries = []
    for operator in df_macro.sOperatore.unique().tolist():

        tmp = df_macro[df_macro.sOperatore == operator].sample(num_ticket, random_state=SEED)
        tickets = tmp.tTicket.str.cat(sep="\n")

        task_prompt  = f"Di seguito ti verranno forniti {num_ticket} ticket di assistenza clienti, uno per riga, risolti dall'operatore {operator}. \n"
        task_prompt += f"Descrivi, con massimo 20 parole, in generale le problematiche risolte dall'operatore. \n"
        task_prompt += f"Non usare condizionale o frasi ipotetiche, ma esprimiti come se fossi sicuro al 100% di quello che stai dicendo. \n"

        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": task_prompt},
                    {"role": "user", "content": tickets}
                ],
                temperature=0,
                top_p=0.01,
                max_tokens=1000
            )

        except openai.error.RateLimitError:
            print("Rate limit reached. Waiting 1 minute...")
            time.sleep(60)
            print("Continuing...")

            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": task_prompt},
                    {"role": "user", "content": tickets}
                ],
                temperature=0,
                top_p=0.01,
                max_tokens=1000
            )

        except Exception:
            print("General Error...")
            time.sleep(30)
            print("Continuing...")

            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": task_prompt},
                    {"role": "user", "content": tickets}
                ],
                temperature=0,
                top_p=0.01,
                max_tokens=1000
            )

        summaries.append(completion.choices[0].message.content)


    summaries_prompt = ""
    for i, s in enumerate(summaries):
        summaries_prompt += f"{i+1}. {s} \n"

    return summaries_prompt


def get_system_prompt(str_operators, str_categories, mode, model, train_df, zero_shot_info="", num_shots=1, num_tickets_summarize=10):

    # Baseline promtp.

    system_prompt  = f"Sei il responsabile di un centro di assistenza che ha il compito di dividere i ticket tra i vari operatori umani.\n"
    system_prompt += f"Gli operatori disponibili, contenuti tra parentesi quadre. sono i seguenti: {str_operators}. \n"
    system_prompt += f"I ticket sono divisi in categorie, elencate con lettere dell'alfabeto, contenute tra parentesi tonde, che sono le seguenti: {str_categories}. \n"
    system_prompt += f"Ogni ticket è composto da una CATEGORIA, un OGGETTO e da una DESCRIZIONE. \n"

    if mode == ZERO_SHOT:

        system_prompt += f"Tieni presente le seguenti informazioni: \n"
        system_prompt += zero_shot_info
        system_prompt += f"Il tuo compito è quello di assegnare il ticket all'operatore più adatto, rispondendo solo con il nome relativo all'operatore. \n"

        print(system_prompt)

        return system_prompt


    elif mode == FEW_SHOT:

        df_shots = pd.read_json(train_df)

        to_drop = []
        for i, r in df_shots.iterrows():
            if r["tTicket"] in df["tTicket"].values:
                to_drop.append(i)

        df_shots.drop(to_drop, inplace=True)
        df_shots.reset_index(drop=True, inplace=True)

        system_prompt += f"Ecco qualche esempio di ticket classificati correttamente: \n\n"
        system_prompt += get_few_shot_prompt(df_shots, num_shots)
        system_prompt += f"Il tuo compito è quello di assegnare il ticket all'operatore più adatto, rispondendo solo con il nome relativo all'operatore. \n"

        return system_prompt
    

    elif mode == SUMMARY:
        print(f"Summarizing tickets...")

        df_summ = pd.read_json(train_df)

        to_drop = []
        for i, r in df_summ.iterrows():
            if r["tTicket"] in df["tTicket"].values:
                to_drop.append(i)

        df_summ.drop(to_drop, inplace=True)
        df_summ.reset_index(drop=True, inplace=True)

        system_prompt += f"Tieni presente le seguenti informazioni: \n"
        system_prompt += summarize_tickets(model, df_summ, num_tickets_summarize)
        system_prompt += f"Il tuo compito è quello di assegnare il ticket all'operatore più adatto, rispondendo solo con il nome relativo all'operatore. \n"

        return system_prompt
    

######## CONSTANTS ########

OPENAI_KEY_FILE = "openai_key.txt"
TESTSET = "datasets/testset_1.json"
TRAIN_DATASET = "datasets/testset_2.json"

SEED = 224964

MODEL = 'gpt-4'

ZERO_SHOT = 0
FEW_SHOT = 1
SUMMARY = 2

MODE = ZERO_SHOT

NUM_SHOTS = 3
NUM_TICKETS_SUMMARIZE = 5


if __name__ == '__main__':

    set_seed(SEED)

    # Initialize the OpenAI API from openai_key.txt.

    with open(OPENAI_KEY_FILE, "r") as f:
        openai.api_key = f.read()

    # Load the dataset
    df = pd.read_json(TESTSET)
    
    # System prompt.
    str_operators = "[" + ", ".join(df.sOperatore.unique().tolist()) + "]"
    str_categories = "(" + ", ".join(df.sCategoria.unique().tolist()) + ")"

    system_prompt = get_system_prompt(str_operators, str_categories, MODE, MODEL, TRAIN_DATASET)


    # Predict the operator.
    predictions = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        ticket_prompt = get_ticket_prompt(row.sCategoria, row.sOggetto, row.tTicket)

        try:
            pred = predict_operator(MODEL, system_prompt, ticket_prompt)

        except openai.error.RateLimitError:
            print("Rate limit reached. Waiting 1 minute...")
            time.sleep(60)
            print("Continuing...")
            pred = predict_operator(MODEL, system_prompt, ticket_prompt)

        except openai.error.ServiceUnavailableError:
            print("The server is overloaded or not ready yet.")
            print("Waiting 30 seconds...")
            time.sleep(30)
            print("Continuing...")
            pred = predict_operator(MODEL, system_prompt, ticket_prompt)

        except Exception:
            print("General error")
            print("Waiting 30 seconds...")
            time.sleep(30)
            print("Continuing...")
            pred = predict_operator(MODEL, system_prompt, ticket_prompt)

        predictions.append(pred)

        time.sleep(3)

    # Results.
    print(classification_report(df.sOperatore, predictions, zero_division=0))
