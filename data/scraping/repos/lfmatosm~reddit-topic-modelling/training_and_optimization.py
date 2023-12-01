import json
import re
from datetime import datetime as dt
from utils.utils import get_best_hyperparameters
from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
from octis.models.CTM import CTM
from octis.models.ETM import ETM
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Integer, Real
import argparse
import os

BASE_OUTPUT_PATH = f'./octis_training/optimizations'

def get_current_datetime():
    return re.sub("\.|:|\ |-", "", f'{dt.now()}')

parser = argparse.ArgumentParser(description='Script for automated LDA, CTM and ETM training/optimization using OCTIS', add_help=True)
parser.add_argument('--dataset_path', type=str, help='base path for the dataset files', required=True)
parser.add_argument('--embeddings_path', type=str, default='empty', help='path for the Word2Vec embeddings files to be used with ETM, if you desire to train this model', required=False)
parser.add_argument('--models', type=str, nargs='+', default=[], help='lowercase space-separated list of models to be trained, e.g.: lda etm ctm', required=False)
parser.add_argument('--all-models', dest='all_models', action='store_true')
parser.add_argument('--min_k', type=int, default=3, help='Minimum no. of topics (k) in the search space, default: 3', required=False)
parser.add_argument('--max_k', type=int, default=30, help='Maximum no. of topics (k) in the search space, default: 30', required=False)
parser.set_defaults(all_models=False)
args = parser.parse_args()

models_to_train = ["lda", "ctm", "etm"] if args.all_models else args.models

if args.all_models or "etm" in args.models:
    assert args.embeddings_path != 'empty', "embeddings_path must be provided to train ETM models!"

dataset = Dataset()
dataset.load_custom_dataset_from_folder(args.dataset_path)

language = args.dataset_path.split(os.path.sep)[-1]

topics_dimension = Integer(low=int(args.min_k), high=int(args.max_k))

models = []
for model_name in models_to_train:
    if model_name == "lda":
        models.append({
            "model": LDA(),
            "search_space": {
                "num_topics": topics_dimension,
            }
        })
    elif model_name == "ctm":
        bert_model = "distiluse-base-multilingual-cased-v1" \
            if language == "pt" else "bert-base-nli-mean-tokens"
        models.append({
            "model": CTM(inference_type="combined", 
                bert_model=bert_model, bert_path=f'./{args.dataset_path.replace("/", "_")}_{language}'),
            "search_space": {
                "num_topics": topics_dimension,
                "lr": Real(2e-3, 2e-1)
            }
        })
    elif model_name == "etm":
        models.append({
            "model": ETM(train_embeddings=False, embeddings_type='keyedvectors', 
                embeddings_path=args.embeddings_path),
            "search_space": {
                "num_topics": topics_dimension,
                "lr": Real(5.0e-3, 5.0e-1)
            }
        })


coherence_metric = Coherence(topk=10, measure="c_npmi", texts=dataset.get_corpus())

for i in range(len(models)):
    optimizer=Optimizer()
    now = get_current_datetime()
    base_results_path = f'{BASE_OUTPUT_PATH}/{now}_{type(models[i]["model"]).__name__}_{language}'
    result = optimizer.optimize(models[i]["model"], dataset, coherence_metric, models[i]["search_space"], 
                                save_path=base_results_path, # path to store the results
                                number_of_call=30, # number of optimization iterations
                                model_runs=5, plot_best_seen=True) # number of runs of the topic model
    #save the results of the optimization in file
    results_path = f'{base_results_path}/{models_to_train[i]}_{language}.json'
    result.save(results_path)
    results_with_best_hyperparams = get_best_hyperparameters(results_path)
    json.dump(results_with_best_hyperparams, open(results_path, "w"), indent=4)
    print(f'Optimization finished. Results can be found at: {results_path}\n\n')
