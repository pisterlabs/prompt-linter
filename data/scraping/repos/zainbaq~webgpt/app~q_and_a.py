import numpy as np
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings
import openai
import argparse
from utils import get_domain, get_openai_key

openai.api_key = get_openai_key()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', '-u', help='url to scrape')
    parser.add_argument('--domain', '-d', help='domain to restrict scrape over')
    return parser.parse_args()

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

def main():
    args = parse_args()
    domain = args.domain
    if args.url and not args.domain:
        domain = get_domain(args.url)

    df = pd.read_csv(f'data/processed/{domain}/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    # df.head()
    print("Session started.\n Enter 'stop' or 'exit' to end the session")
    while True:
        question = input('Question >> ')
        if question == 'stop' or question == 'exit':
            break
        answer = answer_question(df, question=question)
        print(f'Answer >> {answer}')
    print('Session ended.')
if __name__ == '__main__':
    main()