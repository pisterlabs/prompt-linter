import argparse
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from data.dataset import get_dataset
from trainer.utils import get_save_num
from evals.measures import get_topic_words_from_files


my_parser = argparse.ArgumentParser()

my_parser.add_argument('topk', type=int, help='Top-k topic words for calculating coherence and diversity.')
my_parser.add_argument('exp_num', type=int, help='Number of experiments to average over.')
my_parser.add_argument('num_seeds', type=int, help='Number of seeds for each experiment.')
my_parser.add_argument('data', type=str, help='Location of the data pickle file.')
my_parser.add_argument('experiment', type=str, help='Location of the experiment directory.')

args = my_parser.parse_args()

topk = args.topk
experiment_num = args.exp_num
num_seeds = args.num_seeds
data_pickle = args.data.rstrip('.pkl').replace('\\', '/')
experiment_dir = args.experiment.replace('\\', '/')

dataset_save_dict = get_dataset(data_pickle)

train_docs = [doc.split() for doc in dataset_save_dict["train"]["preprocessed_docs"]]
if dataset_save_dict["test"]["preprocessed_docs"] is not None:
    test_docs = [doc.split() for doc in dataset_save_dict["test"]["preprocessed_docs"]]
    docs = train_docs + test_docs
else:
    docs = train_docs

vocabulary = dataset_save_dict["vocabulary"]
dictionary = Dictionary()
dictionary.token2id = vocabulary["token2id"]
dictionary.id2token = vocabulary["id2token"]

true_experiment_num = get_save_num(experiment_num)

total_score = 0
for seed_num in tqdm(range(num_seeds)):
    true_seed_num = get_save_num(seed_num)
    seed_pickle = f"{experiment_dir}/{true_experiment_num}/seeds/{true_seed_num}/{true_seed_num}_plotting_arrays"
    topics = get_topic_words_from_files(data_pickle, seed_pickle, topk=topk)

    npmi_window = CoherenceModel(
        topics=topics,
        texts=docs,
        dictionary=dictionary,
        coherence="c_npmi",
        topn=topk
    )

    score = npmi_window.get_coherence()
    total_score += score

average_score = total_score / num_seeds
print(f"NPMI Coherence (window=10):\n{score}\n")
