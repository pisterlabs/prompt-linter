from token_2 import TOKEN
import cohere


co = cohere.Client(TOKEN)

def embed_text(text):
    output = co.embed(model='small', texts=text)
    return output.embeddings

def classify(text):
    t = [text]
    classifications = co.classify(
    model='ff3b9944-21b6-41de-bf58-dd5e4bd02993-ft',
    inputs=t
    )
    return classifications.classifications[0].prediction