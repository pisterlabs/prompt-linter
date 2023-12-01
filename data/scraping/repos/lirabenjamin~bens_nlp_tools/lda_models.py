import pandas as pd
import re
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from sklearn.model_selection import train_test_split
import os

def run_topic_model(data, text_column, output_folder, sample=False, sample_size=200, n_topics=50):
    # Sampling
    if sample:
        data = data.sample(n=sample_size)
    
    # Filtering based on character count
    data = data[data[text_column].str.len() > 50]

    # Convert text, add spaces after punctuations, and remove numbers 
    data[text_column] = data[text_column].str.replace(".", ". ", regex=False)
    data[text_column] = data[text_column].str.replace(",", ", ", regex=False)
    data[text_column] = data[text_column].str.replace(r"[0-9]+", " ")

    # Process data and create a dictionary and corpus for LDA
    texts = data[text_column].str.split().tolist()
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Conduct LDA
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, random_state=325, passes=20, alpha='auto', eta='auto')

    # Visualization
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(f"{output_folder}/ldavis.html", "w") as f:
        pyLDAvis.save_html(vis, f)

    # Save environment (Not a direct equivalent of R's save.image, but this will save the model)
    lda_model.save(f"{output_folder}/lda_model")

# Example usage:
data = pd.read_csv("IMDB Dataset.csv")
data = data.sample(n=1000)
run_topic_model(data, 'review', 'output_folder', n_topics=3)
