# -------------------------------------------------------------
# stop_list_iter.py
#
# Can be used to run LDA with a specified value of k
# to create a stop list of uninformative terms.
#
# Terms are added to the stop list if they:
#   1. Appear as a top-15 word in many topics.
#   2. Do not carry semantic information upon manual
#      inspection (e.g., "uh", "le", "ca", "am").
#
# This allows us to remove uninformative terms before creating
# the final model.
# -------------------------------------------------------------
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from coherence import Coherence
from collections import defaultdict
import csv
import time
import argparse
import os

from DataManager import DataManager

# Project-wide constants, file paths, etc.
import settings

parser = argparse.ArgumentParser(
    description='Performs k-fold cross validation using LDA on the Twitter dataset, to tune the number of topics K using likelihood as the evaluation metric.'
    )

parser.add_argument('--train_path', type=str, nargs='?', default = "../TwitterDataset/data/Jan27-Feb02/",
                    help='the path to the training file, defaults to ../TwitterDataset/data/Jan27-Feb02/')


parser.add_argument('--output_path', type=str, nargs='?', default = "outwords.temp",
                    help='the path to the training file, defaults to outwords.temp')

def find_candidate_stop_words(top_words):
    """
    For each word, counts the number of topics in which it appears.
    """
    counts = defaultdict(int)
    for t,words in top_words.items():
        for w in words:
            counts[w] += 1

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print("Candidates for Stop List:")
    for w, c in sorted_counts:
        if c >= 2:
            print("   -", w, c)


def main():
    '''
    Driver code for the project.
    '''
    args = parser.parse_args()

    # Extract the data for LDA and divide into 10 folds
    dm = DataManager(args.train_path, 'twitter')
    print("Loading data...")

    if os.path.exists("tweet_cache.cache"):
        os.system("rm tweet_cache.cache")
    start = time.perf_counter()
    dm.load_data("tweet_cache.cache")
    end = time.perf_counter()
    if settings.DEBUG: print(f"Preparing the data (loading, normalizing) took {end-start:0.4f} seconds.")

    print("Training word2vec...")
    coh = Coherence()
    coh.mapWordsToVecs(dm.get_all_data())

    # trying a bunch of values of k to compare the coherence
    print("Training LDA model:")
    start = time.perf_counter()
    # Train the model with the param choice.
    transformed, model, vectorizer = dm.run_lda(num_components=10)
    # Compute the resulting accuracy on the validation set.
    end = time.perf_counter()
    if settings.DEBUG: print(f"        Training took {end-start:0.4f} seconds.")

    print("Finding top words:")
    top_words = dm.get_top_words_per_topic(model, vectorizer, 15)
    print("TOP WORDS:")
    for t, words in top_words.items():
        print(f"    {t}: {words}")

    find_candidate_stop_words(top_words)

    print("Finding coherence of each topic:")
    coh_list = []
    for topic in top_words:
        topic_coherence = coh.getCoherence(top_words[topic])
        # print(topic, topic_coherence)
        coh_list.append(topic_coherence)
    avg_coh = sum(coh_list) / len(coh_list)
    print("    Average Coherence =", avg_coh)

    print("Storing words to output...")
    dm.save_words_as_json(top_words, args.output_path)


# Entry point to the cross validation (LDA) program.
if __name__ == '__main__':
    main()

