import argparse
import sys

# Add src to path and make imports
sys.path.append('../..')

from src.evaluateNN.OCTIS.octis.evaluation_metrics.diversity_metrics import (
    InvertedRBO, TopicDiversity)
from skopt.space.space import Categorical, Integer, Real
from src.evaluateNN.OCTIS.octis.optimization.optimizer import Optimizer
from src.evaluateNN.OCTIS.octis.models.ProdLDA import ProdLDA
from src.evaluateNN.OCTIS.octis.models.CTM import CTM
from src.evaluateNN.OCTIS.octis.evaluation_metrics.coherence_metrics import Coherence
from src.evaluateNN.OCTIS.octis.dataset.dataset import *

def evaluate(corpus: str,
             output_folder: str,
             optimization_runs: int = 30,
             model_runs: int = 1) -> None:
    """Evaluate the CTM model on the given corpus.

    Parameters:
    ----------
        corpus (str): Path to the corpus.
        output_folder (str): Path where OCTIS data will be saved.
        optimization_runs (int, optional): Number of optimization runs. Defaults to 30.
        model_runs (int, optional): Number of model runs. Defaults to 1.

    Returns:
    -------
        None
        Results are saved at output_folder + "/results".
    """

    # Create dataset in OCTIS format
    dataset = Dataset()
    dataset.load_custom_dataset_from_parquet(
        path_to_parquet=corpus,
        path_to_octis_data=output_folder)

    # Load topic model
    model = CTM(num_topics=10, num_epochs=30, inference_type='combined',
                bert_model="bert-base-nli-mean-tokens")

    # Define evaluation metric
    npmi = Coherence(texts=dataset.get_corpus())
    cv = Coherence(texts=dataset.get_corpus(), measure='c_v')
    td = TopicDiversity(topk=10)
    irbo = InvertedRBO(topk=10)

    # Define hyperparameters search space
    search_space = {"activation": Categorical({'sigmoid', 'relu',
                                               'softplus', 'tanh'}),
                    "solver": Categorical({'adagrad', 'adam',
                                           'sgd', 'adadelta', 'rmsprop'}),
                    "dropout_thetas": Real(0.0, 0.95),
                    "dropout_inf": Real(0.0, 0.95),
                    "lr": Real(1e-4, 1e-2),
                    "momentum": Real(0.0, 0.99),
                    }

    optimizer = Optimizer()
    optimization_result = optimizer.optimize(
        model, dataset, npmi, search_space, number_of_call=optimization_runs,
        model_runs=model_runs, save_models=True,
        extra_metrics=[td, irbo, cv],  # to keep track of other metrics
        save_path=output_folder+"/results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_corpus', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/cordis_lemmas_embeddings.parquet",
                        help="Path to the corpus.")
    parser.add_argument('--octis_folder', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/exp_octis",
                        help="Path where OCTIS data will be saved.")
    parser.add_argument('--optimization_runs', type=int,
                        default=90,
                        help="Number of optimization runs.")
    parser.add_argument('--model_runs', type=int,
                        default=1,
                        help="Number of model runs.")
    args = parser.parse_args()

    evaluate(corpus=args.path_corpus,
             output_folder=args.octis_folder,
             optimization_runs=args.optimization_runs,
             model_runs=args.model_runs)


if __name__ == "__main__":
    main()
