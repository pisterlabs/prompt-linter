import pandas as pd
import numpy as np
import pandarallel
import cohere

co = cohere.Client('<key>')

pandarallel.pandarallel.initialize(progress_bar=False)

def load_embeddings(filename):
    """
    Load the embeddings from a CSV file.
    """

    df = pd.read_csv(filename)
    df.reset_index(drop=True, inplace=True)
    df = df.dropna()

    # embeddings = np.array(df['embeddings'].parallel_map(lambda x: eval(x)).tolist())
    embeddings = np.load('/mnt/d/embs.npy')

    return df, embeddings

# Pre-load the embeddings at startup
song_df, song_embeddings = load_embeddings('new_music_embeddings.csv')


def get_topk(song_summary, embeddings, k, model = 'english'):
    """
    Get the top k most similar songs to the given song summary.
    """

    song_embedding = co.embed([song_summary], model=f'embed-{model}-v2.0').embeddings[0]
    distances = np.linalg.norm(embeddings - song_embedding, axis=1)
    top = distances.argsort()[:k]
    return top


def retrieve_song(song_data, k=10):
    """
    Retrieve a song from the database.
    """

    topk = get_topk(song_data, song_embeddings, k)
    resp = dict(song_df.iloc[topk[np.random.randint(k)]])
    resp = {
        'track_name': resp['track_name'],
        'artist_name': resp['artist_name'],
        'lyrics': resp['lyrics'],
    }
    return resp


def regenerate_song(song_data):
    """
    Since the lyrics in the database are automatically generated, we need to regenerate them.
    We do that by leveraging the cohere Chat model with Retrieval-Augmented Generation.
    """

    # Retrieve the low-quality song lyrics from the database
    song_data = retrieve_song(song_data)

    # Ask the chat model to regenerate the song lyrics
    output = co.chat(
        f"Write the lyrics of the song \"{song_data['track_name']}\" by \"{song_data['artist_name']}\". Start with the word BEGIN and end with the word END.",
        connectors = [{"id": "web-search"}]
    ).text

    # If the chat model fails to generate the lyrics, return a failed message
    if not ('BEGIN' in output and 'END' in output):
        return {"track_name": song_data['track_name'], "artist_name": song_data['artist_name'], "lyrics": "failed"}

    # Otherwise, parse the output and return it
    output = output.split('BEGIN')[1].split('END')[0].strip()
    output = output.replace('\n', '<br>')

    return {"track_name": song_data['track_name'].title(), "artist_name": song_data['artist_name'].title(), "lyrics": output}




