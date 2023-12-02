
import pandas as pd
import openai
import multiprocessing as mp
from itertools import repeat
import backoff



@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Embedding.create(**kwargs)

def openai_embed_text(text, api_key, model = "text-embedding-ada-002"):
    """
    
    Parameters:
    -----------

    Returns:
    --------
    """
    try:
        openai.api_key = api_key
        response = completions_with_backoff(
            input=text,
            model=model,
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(e)
        return None


def main():
    api_key = ""
    with open("../results/schembl_summs_v3_alg_cleaned_vocab.txt") as f:
        labels = f.read().splitlines()

    df = pd.DataFrame(labels, columns = ["labels"])

    n_cpus = 16
    with mp.Pool(n_cpus) as p:
        embeddings = p.starmap(openai_embed_text, zip(labels, repeat(api_key)))

    df['embeddings'] = embeddings

    df.to_pickle("../results/schembl_summs_v3_vocab_embeddings.pkl")
    df.to_csv("../results/schembl_summs_v3_vocab_embeddings.csv")

if __name__ == "__main__":
    main()