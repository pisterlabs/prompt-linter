
# run_nmf_twitter.py
#
# Can be used to run NMF with a specified value of k,
# and reports the coherence. Final topics are saved to the 
# results path.
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
    description='Runs NMF on the twitter data, reports the resulting coherence, and prints the results to the output path.'
    )

parser.add_argument('--train_path', type=str, nargs='?', default = "../TwitterDataset/data/Jan27-Feb02/",
                    help='the path to the twitter dir, defaults to ../TwitterDataset/data/Jan27-Feb02/')


parser.add_argument('--output_path', type=str, nargs='?', default = "outwords.temp",
                    help='the path to the results file, defaults to outwords.temp')

def main():
    '''
    Driver code for the project.
    '''
    args = parser.parse_args()

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
    print("Training NMF model:")
    start = time.perf_counter()
    # Train the model with the param choice.
    transformed, model, vectorizer = dm.run_nmf(num_components=10)
    # Compute the resulting accuracy on the validation set.
    end = time.perf_counter()
    if settings.DEBUG: print(f"        Training took {end-start:0.4f} seconds.")

    print("Finding top words:")
    top_words = dm.get_top_words_per_topic(model, vectorizer, 15)
    print("TOP WORDS:")
    for t, words in top_words.items():
        print(f"    {t}: {words}")

    print("Finding coherence of each topic:")
    coh_list = []
    for topic in top_words:
        topic_coherence = coh.getCoherence(top_words[topic])
        print(topic, topic_coherence)
        coh_list.append(topic_coherence)
    avg_coh = sum(coh_list) / len(coh_list)
    print("    Average Coherence =", avg_coh)

    print("Storing words to output...")
    dm.save_words_as_json(top_words, args.output_path)


# Entry point to the run NMF program.
if __name__ == '__main__':
    main()

