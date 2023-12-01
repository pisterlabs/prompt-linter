import gensim
from gensim.models.coherencemodel import CoherenceModel
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import gensim.corpora as corpora
import argparse

def calculate_coherence_score(topic_model, docs):
    """
    Calculate the coherence score of a topic model.

    Parameters:
    topic_model (BERTopic): The BERTopic model whose coherence is to be calculated.
    docs (list of str): The documents used in the topic model.

    Returns:
    float: The coherence score of the topic model.
    """
    # Vectorizer model from the topic model
    cv = topic_model.vectorizer_model

    # Tokenize the documents
    doc_tokens = [text.split(" ") for text in docs]

    # Create a Gensim Dictionary and Corpus
    id2word = corpora.Dictionary(doc_tokens)
    texts = doc_tokens
    corpus = [id2word.doc2bow(text) for text in texts]

    # Extract topic words
    topic_words = []
    for i in range(len(topic_model.get_topic_freq())-1):
        interim = [t[0] for t in topic_model.get_topic(i)]
        topic_words.append(interim)

    # Calculate and return coherence score
    coherence_model = CoherenceModel(topics=topic_words, texts=texts, corpus=corpus, dictionary=id2word, coherence='c_v')
    return coherence_model.get_coherence()

def topic_diversity(words_per_topics):
    """
    Calculate the topic diversity of a topic model.

    Parameters:
    words_per_topics (list of list of str): The words in each topic.

    Returns:
    float: The topic diversity score.
    """
    unique_words = set()
    total_words = 0
    for words in words_per_topics:
        unique_words.update(words)
        total_words += len(words)
    return len(unique_words) / total_words

def main(model, texts):
    """
    Main function to calculate and print the topic coherence and diversity scores.

    Parameters:
    model (BERTopic): The BERTopic model to evaluate.
    texts (list of str): The documents used in the topic model.
    """
    # Calculate Topic Coherence
    coherence_score = calculate_coherence_score(model, texts)
    print(f"Topic Coherence Score: {coherence_score}")

    # Calculate Topic Diversity
    diversity_score = topic_diversity(model.get_topic_info().Representation.to_list())
    print(f"Topic Diversity Score: {diversity_score}")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Evaluate BERTopic model")

    # Add argument for model path or repository ID
    parser.add_argument("--path_or_repoID", help="Path or repository ID of the BERTopic model", type=str)

    # Parse the arguments
    args = parser.parse_args()

    # Load texts
    texts = pd.read_csv("src/frontend/data/df_telegram.csv")["messageText"].tolist()

    # Call the main function with parsed arguments
    main(args.path_or_repoID, texts)
