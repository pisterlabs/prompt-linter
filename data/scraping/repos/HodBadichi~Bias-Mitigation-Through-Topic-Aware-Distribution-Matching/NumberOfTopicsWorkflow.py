import logging
import csv
import os
import sys

if os.name != 'nt':
    sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

import numpy as np
import gensim
import pandas as pd
from gensim.models import CoherenceModel
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

from TopicModeling.Utils.Metrics import LDAMetrics
from TopicModeling.LDA.src.LDAUtils import GetLDAParams, PrepareData
from TopicModeling.Utils.TopcModelingUtils import getCurrRunTime
from TopicModeling.LDA.src.hparams_config import hparams

"""
Workflow for tuning gensim`s LDA model.
"""


def ShowEvaluationGraphs(
        train_results_path,
        test_results_path,
        smooth=False,
        poly_deg=None
):
    """
    Used for 'NumberOfTopicsExperiment' results.

    Print on the screen metrics results against number of topics
    :param file_path:CSV file which holds in a single column 'Number of topics' and different measures
    :param dataset_name: string, `test` or `train`
    :param smooth: Boolean flag, if True it smooth the results graph
    :param poly_deg: Int, matches a polynomial of degree 'poly_deg'  to the results graph
    :return:None
    """
    figure, axis = plt.subplots(2, 3)
    figure.set_size_inches(18.5, 10.5)
    train_evaluation_csv = pd.read_csv(train_results_path)
    test_evaluation_csv = pd.read_csv(test_results_path)
    column_names = train_evaluation_csv.columns

    for measure, ax in zip(column_names, axis.ravel()):
        if measure == 'Topics':
            continue
        train_scores = train_evaluation_csv[measure].tolist()
        test_scores = test_evaluation_csv[measure].tolist()

        train_X_Y_Spline = make_interp_spline(train_evaluation_csv.Topics.tolist(), train_scores)
        test_X_Y_Spline = make_interp_spline(test_evaluation_csv.Topics.tolist(), test_scores)

        # Returns evenly spaced numbers
        # over a specified interval.
        train_X_ = np.linspace(train_evaluation_csv.Topics.min(), train_evaluation_csv.Topics.max(), 500)
        train_Y_ = train_X_Y_Spline(train_X_)
        test_X_ = np.linspace(test_evaluation_csv.Topics.min(), test_evaluation_csv.Topics.max(), 500)
        test_Y_ = test_X_Y_Spline(test_X_)
        if poly_deg is not None:
            train_coefs = np.polyfit(train_evaluation_csv.Topics.tolist(), train_scores, poly_deg)
            train_y_poly = np.polyval(train_coefs, train_evaluation_csv.Topics.tolist())
            ax.plot(train_evaluation_csv.Topics.tolist(), train_scores, "o", label="data points")
            ax.plot(train_evaluation_csv.Topics, train_y_poly, label="Train", color='blue')

            test_coefs = np.polyfit(test_evaluation_csv.Topics.tolist(), test_scores, poly_deg)
            test_y_poly = np.polyval(test_coefs, test_evaluation_csv.Topics.tolist())
            ax.plot(test_evaluation_csv.Topics.tolist(), test_scores, "o", label="data points")
            ax.plot(test_evaluation_csv.Topics, test_y_poly, label="Test", color='red')
        elif smooth is False:
            ax.plot(train_evaluation_csv.Topics, train_scores, label="Train", color='blue')
            ax.plot(test_evaluation_csv.Topics, test_scores, label="Test", color='red')
        else:
            ax.plot(train_X_, train_Y_, label="Train", color='blue')
            ax.plot(test_X_, test_Y_, label="Test", color='red')

        ax.set_title(measure + " Measure ")
        ax.set_xlabel("Number of topics")
        ax.set_ylabel("Measure values")
        plt.savefig(os.path.join(os.pardir, 'results', f'eval_graphs_{getCurrRunTime()}'))
    plt.close()

def InitializeLogger(sFileName=None):
    """
    Intialize a global logger to document the experiment
    :param sFileName: required logger name
    :return: None
    """
    logs_directory_path = os.path.join(os.pardir, 'logs')
    os.makedirs(logs_directory_path, exist_ok=True)

    if sFileName is None:
        sFileName = f'log_{getCurrRunTime()}.txt'
    LoggerPath = os.path.join(os.pardir, 'logs', sFileName)
    from importlib import reload
    reload(logging)
    logging.basicConfig(
        filename=LoggerPath,
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.NOTSET)


def RunTuningProcess(
        train_LDA_parameters,
        test_LDA_parameters,
        topics_range,
        passes=1,
        iterations=1,
        chunksize=300,
):
    """
    Train and Evaluate each model with a different number of topics by the topics range given from the input.
    each model is saved and evaluated by 'perplexity' and 'Coherence' measurements, when the evaluation results
    are saved as well.
    :param train_LDA_parameters: dictionary of {'corpus','texts','id2word'}
    :param test_LDA_parameters: dictionary of {'corpus','texts','id2word'}
    :param topics_range: topics range to tune
    :param passes: number of passes the model does on the whole corpus
    :param iterations: number of iterations the model does
    :param chunksize: number of documents in each iteration
    :return: None
    """
    params_dictionary = {
        'passes': passes,
        'iterations': iterations,
        'chunksize': chunksize,
        'topics_range': topics_range
    }
    logging.info(params_dictionary)

    models_directory_path = os.path.join(os.pardir, 'saved_models')
    result_directory_path = os.path.join(os.pardir, 'results')
    os.makedirs(models_directory_path, exist_ok=True)
    os.makedirs(result_directory_path, exist_ok=True)

    # Keys are meant to write CSV headers later on ,values are dummy values
    # my_dict = {"Topics": 6, "u_mass": 5, "c_uci": 4, "c_npmi": 3, "c_v": 2, "perplexity": 1}
    field_names = ['Topics', 'u_mass', 'c_uci', 'c_npmi', 'c_v', 'perplexity']
    train_results_path = os.path.join(result_directory_path, fr'train_evaluation_{getCurrRunTime()}.csv')
    test_results_path = os.path.join(result_directory_path, fr'test_evaluation_{getCurrRunTime()}.csv')
    with open(train_results_path, "w", encoding='UTF8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()

    with open(test_results_path, "w", encoding='UTF8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()

    for num_of_topics in topics_range:
        logging.info(f"Running number of topics model : {num_of_topics}")
        curr_lda_model = gensim.models.ldamodel.LdaModel(
            corpus=train_LDA_parameters['corpus'],
            id2word=train_LDA_parameters['id2word'],
            num_topics=num_of_topics,
            random_state=42,
            update_every=1,
            chunksize=chunksize,
            passes=passes,
            iterations=iterations
        )

        saved_model_path = os.path.join(models_directory_path, f'model_{num_of_topics}_{getCurrRunTime()}')
        curr_lda_model.save(saved_model_path)
        # Save results of train_dataset set

        with open(train_results_path, "a", encoding='UTF8', newline='') as csv_file:
            # Initialize 'LDAMetrics' class
            my_metrics = LDAMetrics(curr_lda_model, train_LDA_parameters['corpus'], train_LDA_parameters['texts'])
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            result_dict = my_metrics.EvaluateAllMetrics()
            result_dict['Topics'] = num_of_topics
            writer.writerows([result_dict])

        with open(test_results_path, "a", encoding='UTF8', newline='') as csv_file:
            # Initialize 'LDAMetrics' class
            my_metrics = LDAMetrics(curr_lda_model, test_LDA_parameters['corpus'], test_LDA_parameters['texts'])
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            result_dict = my_metrics.EvaluateAllMetrics()
            result_dict['Topics'] = num_of_topics
            writer.writerows([result_dict])


def RunNumberOfTopicsExperiment():
    np.random.seed(42)

    InitializeLogger()
    train_set, test_set = PrepareData()

    #   Gensim LDA preparation - Create corpus and id2word
    train_LDA_params = GetLDAParams(train_set)
    test_LDA_params = GetLDAParams(test_set)

    RunTuningProcess(
        train_LDA_params,
        test_LDA_params,
        topics_range=hparams['topics_range'],
        passes=hparams['passes'],
        iterations=hparams['iterations'],
        chunksize=hparams['chunksize'],
    )
    logging.info("Evaluating stage is done successfully ")

    test_results_path = os.path.join(os.pardir, 'results', fr'test_evaluation_{getCurrRunTime()}.csv', )
    train_results_path = os.path.join(os.pardir, 'results', fr'train_evaluation_{getCurrRunTime()}.csv')

    ShowEvaluationGraphs(train_results_path,test_results_path, True, None)


if __name__ == '__main__':
    # to run on Technion lambda server use: sbatch -c 2 --gres=gpu:1 run_on_server.sh -o run.out
    RunNumberOfTopicsExperiment()
