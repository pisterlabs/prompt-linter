# Import libraries

import os
import sys
from useful_fun import HiddenPrints
from custom.metrics.TDCI import TDCI
from octis.models.CTM import CTM
from octis.models.ETM import ETM
from octis.models.HDP import HDP
from octis.models.LDA import LDA
from octis.models.LSI import LSI
from octis.models.NMF import NMF
from octis.models.ProdLDA import ProdLDA
from octis.dataset.dataset import Dataset
from octis.models.NeuralLDA import NeuralLDA
from custom.models import CustomTop2Vec, CustomBERTopic
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Categorical, Integer
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

base_csv_path = "tunning/csv/"
base_res_path = "tunning/res/"
dataset_folder_path = ""
dataset_name = ""


def __optimize(model: any, search_space: dict, save_path: str, csv_path: str):
    """
    Optimize a given model in a search space.

    """
    # Create optimizer
    optimizer = Optimizer()

    with HiddenPrints():
        # Optimize
        optimization_result = optimizer.optimize(
            model,
            dataset,
            tdci,
            search_space,
            random_state=42,
            model_runs=model_runs,
            save_models=True,
            # to keep track of other metrics
            extra_metrics=[topic_coherence, topic_diversity],
            number_of_call=optimization_runs,
            save_path=save_path
        )

        # Export csv
        optimization_result.save_to_csv(csv_path)


def optimize_ctm(n_topics: int = 10):
    # Define search space
    ctm_search_space = {
        "num_topics": [n_topics],
        "model_type": Categorical({"LDA", "prodLDA"}),
        "num_layers": Categorical({1, 2, 3, 4}),
        "num_neurons": Categorical({10, 30, 50, 100, 200, 300}),
        "activation": Categorical(
            {"softplus", "relu", "sigmoid", "tanh",
                "leakyrelu", "rrelu", "elu", "selu"}
        ),
        "solver": Categorical({"adam", "sgd"}),
        "dropout": Real(0.0, 0.95),
        "inference_type": Categorical({"zeroshot", "combined"}),
        "num_epochs": Integer(100, 500),
        "num_samples": Integer(10, 50)
    }

    # Define model and optimize
    model_ctm = CTM()
    __optimize(
        model_ctm,
        ctm_search_space,
        base_res_path + "ctm" + "-" +
        dataset_name + "-" + str(n_topics) + "//",
        base_csv_path + "ctm" + "-" + dataset_name +
        "-" + str(n_topics) + ".csv"
    )


def optimize_etm(n_topics: int = 10):
    etm_search_space = {
        "num_topics": [n_topics],
        "optimizer": Categorical({"adam", "adagrad", "adadelta", "rmsprop", "asgd", "sgd"}),
        "t_hidden_size": Integer(400, 1000),
        "activation": Categorical({'sigmoid', 'relu', 'softplus'}),
        "dropout": Real(0.0, 0.95),
        "num_epochs": Integer(100, 500)
    }

    model_etm = ETM(device="gpu")
    __optimize(model_etm,
               etm_search_space,
               base_res_path + "etm" + "-" +
               dataset_name + "-" + str(n_topics) + "//",
               base_csv_path + "etm" + "-" + dataset_name +
               "-" + str(n_topics) + ".csv"
               )


def optimize_lda(n_topics: int = 10):
    lda_search_space = {
        "num_topics": [n_topics],
        "chunksize": Integer(500, 50000),
        "passes": Integer(1, 500),
        "alpha": Categorical({"symmetric", "asymmetric"}),
        "decay": Real(0.5, 1),
        "offset": Real(0.1, 10),
        "iterations": Integer(5, 1000),
        "gamma_threshold": Real(0.001, 1)
    }

    model_lda = LDA()
    __optimize(model_lda,
               lda_search_space,
               base_res_path + "lda" + "-" +
               dataset_name + "-" + str(n_topics) + "//",
               base_csv_path + "lda" + "-" + dataset_name +
               "-" + str(n_topics) + ".csv"
               )


def optimize_lsi(n_topics: int = 10):
    lsi_search_space = {
        "num_topics": [n_topics],
        "chunksize": Integer(500, 50000),
        "decay": Real(0.5, 1),
        # "power_iters": Integer(1, 10),
        "extra_samples": Integer(10, 200)
    }

    model_lsi = LSI()
    __optimize(model_lsi,
               lsi_search_space,
               base_res_path + "lsi" + "-" +
               dataset_name + "-" + str(n_topics) + "//",
               base_csv_path + "lsi" + "-" + dataset_name +
               "-" + str(n_topics) + ".csv"
               )


def optimize_nmf(n_topics: int = 10):
    nmf_search_space = {
        "num_topics": [n_topics],
        "chunksize": Integer(500, 10000),
        "passes": Integer(1, 500),
        "kappa": Real(0.5, 1),
        "minimum_probability": Real(0.01, 0.1),
        "w_max_iter": Integer(100, 500),
        "w_stop_condition": Real(0.0001, 0.001),
        "h_max_iter": Integer(10, 100),
        "h_stop_condition": Real(0.001, 0.01)
    }

    model_nmf = NMF()
    __optimize(model_nmf,
               nmf_search_space,
               base_res_path + "nmf" + "-" +
               dataset_name + "-" + str(n_topics) + "//",
               base_csv_path + "nmf" + "-" + dataset_name +
               "-" + str(n_topics) + ".csv"
               )


def optimize_prodlda(n_topics: int = 10):
    prodlda_search_space = {
        "num_topics": [n_topics],
        "activation": Categorical({"softplus", "relu"}),
        "dropout": Real(0.0, 0.95),
        "lr": Real(0.0001, 0.01),
        "momentum": Real(0.5, 0.99),
        "solver": Categorical({"adam", "sgd"}),
        "num_epochs": Integer(50, 500),
        "num_layers": Integer(1, 10),
        "num_neurons": Integer(10, 1000),
        "num_samples": Integer(5, 50)
    }

    model_prodlda = ProdLDA()
    __optimize(model_prodlda,
               prodlda_search_space,
               base_res_path + "prodlda" + "-" +
               dataset_name + "-" + str(n_topics) + "//",
               base_csv_path + "prodlda" + "-" +
               dataset_name + "-" + str(n_topics) + ".csv"
               )


def optimize_neurallda(n_topics: int = 10):
    neurallda_search_space = {
        "num_topics": [n_topics],
        "activation": Categorical({"softplus", "relu"}),
        "dropout": Real(0.0, 0.95),
        "lr": Real(0.0001, 0.01),
        "momentum": Real(0.5, 0.99),
        "solver": Categorical({"adam", "sgd"}),
        "num_epochs": Integer(50, 500),
        "num_layers": Integer(1, 10),
        "num_neurons": Integer(10, 1000),
        "num_samples": Integer(5, 50)
    }

    model_neurallda = NeuralLDA()
    __optimize(model_neurallda,
               neurallda_search_space,
               base_res_path + "neurallda" + "-" +
               dataset_name + "-" + str(n_topics) + "//",
               base_csv_path + "neurallda" + "-" +
               dataset_name + "-" + str(n_topics) + ".csv"
               )


def optimize_hdp():
    hdp_search_space = {
        "kappa":  Real(0.5, 1),
        "tau": Real(8, 128),
    }

    model_hdp = HDP()
    __optimize(model_hdp,
               hdp_search_space,
               base_res_path + "hdp" + "-" + dataset_name + "//",
               base_csv_path + "hdp" + "-" + dataset_name + ".csv"
               )


def optimize_custom_top2vec():
    custom_top2vec_search_space = {
    }

    model_custom_top2vec = CustomTop2Vec()
    __optimize(model_custom_top2vec,
               custom_top2vec_search_space,
               base_res_path + "top2vec" + "-" + dataset_name + "//",
               base_csv_path + "top2vec" + "-" + dataset_name + ".csv"
               )


def optimize_bertopic():
    bertopic_search_space = {
    }

    model_bertopic = CustomBERTopic()
    __optimize(model_bertopic,
               bertopic_search_space,
               base_res_path + "bertopic" + "-" + dataset_name + "//",
               base_csv_path + "bertopic" + "-" + dataset_name + ".csv"
               )


def rm_trash():
    if os.path.exists("_test.pkl"):
        os.remove("_test.pkl")

    if os.path.exists("_train.pkl"):
        os.remove("_train.pkl")

    if os.path.exists("_val.pkl"):
        os.remove("_val.pkl")


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Usage: python3 tunning.py dataset_folder_path")
        exit(1)

    # os.chdir(os.getenv("HOME"))
    # os.chdir("octis-compair")
    dataset_folder_path = sys.argv[1]
    dataset_name = dataset_folder_path.split("/")[-1]

    model_runs = 3
    optimization_runs = 10

    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(dataset_folder_path)

    # Define metric for tunning
    topic_coherence = Coherence(texts=dataset.get_corpus())

    # Define other metrics
    topic_diversity = TopicDiversity(topk=10)

    tdci = TDCI(texts=dataset.get_corpus())

    # optimization_topic_functions = [
    #                                 optimize_ctm,
    #                                 optimize_etm,
    #                                 optimize_lda,
    #                                 optimize_lsi,
    #                                 optimize_nmf,
    #                                 optimize_prodlda,
    #                                 optimize_neurallda
    #                                 ]

    # # Run optimizer for models depending on n_topics
    # for n_topics in [10, 20, 30, 40, 50]:
    #     print("Optimizing for n_topics = " + str(n_topics) + "...")
    #     for func in optimization_topic_functions:
    #         with HiddenPrints():
    #             func(n_topics)
    #             rm_trash()
    #         print(str(func.__name__) + " done!")

    optimize_hdp()
