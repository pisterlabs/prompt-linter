"""
Trains topic models for any range of topics and chooses best model based on computed c_v coherence
"""
import argparse
import pandas as pd
import compress_json
import redditcleaner
import os
import little_mallet_wrapper as lmw

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from matplotlib import pyplot as plt

from topic_utils import get_all_chunks_from_column

from text_utils import remove_emojis, process_s, split_story_10, clean_training_text

def get_args():
    parser = argparse.ArgumentParser("Train topic models and choose the best model based on c_v coherence score")
    #df with all birth stories
    parser.add_argument("--birth_stories_df", default="/home/daphnaspira/birthing_experiences/src/birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    #for topic_modeling
    parser.add_argument("--path_to_mallet", default="/home/daphnaspira/birthing_experiences/src/mallet-2.0.8/bin/mallet", help="path where mallet is installed", type=str)
    parser.add_argument("--path_to_save", default="Topic_Modeling/output", help="output path to store topic modeling training data", type=str)
    parser.add_argument("--output_coherence_plot", default="/home/daphnaspira/birthing_experiences/data/Topic_Modeling_Data/topic_coherences.png", help="output path to store line plot of coherence scores")
    parser.add_argument("--output_coherence_csv", default="/home/daphnaspira/birthing_experiences/data/Topic_Modeling_Data/topic_coherence_df.csv", help="output path to store csv of coherence scores")
    parser.add_argument("--start", default=5, help="start value for range of numbers of topics to train the model on")
    parser.add_argument("--stop", default=51, help="stop value for range of numbers of topics to train the model on")
    parser.add_argument("--step", default=1, help="step value for range of numbers of topics to train the model on")
    args = parser.parse_args()
    print(args)
    return args

def split_data(birth_stories_df):
	#split data for training
	birth_stories_df['10 chunks'] = birth_stories_df['Cleaned Submission'].apply(split_story_10)

	#makes list of all chunks to input into LMW
	training_chunks = get_all_chunks_from_column(birth_stories_df['10 chunks'])

	return training_chunks

def lmw_coherence(topic_keys, training_data):
    #Computes c_v coherence from LMW model using Gensim

    #load topics from topic keys

    #data (texts) from training_data
    data = pd.DataFrame(training_data)
    #data.reset_index(drop=True, inplace=True)

    #format for coherence model
    data = data.apply(clean_training_text)
    data = list(data[0])

    #tokenize for coherence model        
    tokens = [string.split() for string in data]

    #make dictionary
    id2word = corpora.Dictionary(tokens)

    #Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(topics=topic_keys, texts=tokens, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    return coherence_ldamallet

def coherence_plot(df, output):
	plt.plot(df)
	plt.title('Topic Coherence per Number of Topics')
	plt.xlabel('Topic Numbers')
	plt.ylabel('Topic Coherence')
	plt.savefig(output)

def main():

	args = get_args()

	#1. prepare data for training topic models
	birth_stories_df = prepare_data(args.birth_stories_df)
	training_data = split_data(birth_stories_df)
	
	#2. for loop:
		#train topic model
		#score the topic model

	coherences = {}
	highest_coherence = (0,0)
	for k in range(args.start, args.stop, args.step):

		if not os.path.exists(f"{args.path_to_save}/{k}"):
			os.mkdir(f"{args.path_to_save}/{k}")

		topic_keys, topic_doc_distributions = lmw.quick_train_topic_model(args.path_to_mallet, f"{args.path_to_save}/{k}", k, training_data)
	
		coherence_score = lmw_coherence(topic_keys, training_data)

		if coherence_score > highest_coherence[0]:
			highest_coherence = (coherence_score, k)

		coherences[k] = coherence_score
	coherence_df = pd.Series(coherences, dtype='float64')
	coherence_df.to_csv(args.output_coherence_csv)

	coherence_plot(coherence_df, args.output_coherence_plot)
	#4. which score had the highest coherence
	print(highest_coherence)

if __name__ == "__main__":
    main()