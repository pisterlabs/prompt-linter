from datetime import datetime
import os
import time

import numpy as np
import psutil
from octis.evaluation_metrics.classification_metrics import F1Score, PrecisionScore, RecallScore, AccuracyScore
from octis.evaluation_metrics.coherence_metrics import Coherence, WECoherencePairwise, WECoherenceCentroid
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO, WordEmbeddingsInvertedRBO, \
    WordEmbeddingsInvertedRBOCentroid
from octis.evaluation_metrics.metrics import AbstractMetric


class DocPerplexity(AbstractMetric):
    def __init__(self):
        super().__init__()

    def score(self, model_output):
        losses = model_output['doc-losses']
        return losses.mean()


class WordPerplexity(AbstractMetric):
    def __init__(self):
        super().__init__()

    def score(self, model_output):
        losses = model_output['word-losses']
        return losses.mean()


Evaluations = {
    'F1': F1Score,
    'Precision': PrecisionScore,
    'Recall': RecallScore,
    'Accuracy': AccuracyScore,
    'NPMI': Coherence,
    'C_V': Coherence,
    'U_MASS': Coherence,
    'C_UCI': Coherence,
    'WECoherencePairwise': WECoherencePairwise,
    'WECoherenceCentroid': WECoherenceCentroid,
    'TopicDiversity': TopicDiversity,
    'InvertedRBO': InvertedRBO,
    'WordEmbeddingsInvertedRBO': WordEmbeddingsInvertedRBO,
    'WordEmbeddingsInvertedRBOCentroid': WordEmbeddingsInvertedRBOCentroid,
    'DocPerplexity': DocPerplexity,
    'WordPerplexity': WordPerplexity
}

Evaluation_params = {
    'F1': {'average': 'macro'},
    'Precision': {'average': 'macro'},
    'Recall': {'average': 'macro'},
    'Accuracy': {'average': 'macro'},
    'NPMI': {'measure': 'c_npmi', 'topk': 10},
    'C_V': {'measure': 'c_v', 'topk': 10},
    'U_MASS': {'measure': 'u_mass', 'topk': 10},    # data_file = '/home/ihussien/Downloads/raw_corpus_converted.txt'.format(
    #     option['dataset'])
    'C_UCI': {'measure': 'c_uci', 'topk': 10},
    'WECoherencePairwise': {'topk': 10},
    'WECoherenceCentroid': {'topk': 10},
    'TopicDiversity': {'topk': 10},
    'InvertedRBO': {'topk': 10},
    'WordEmbeddingsInvertedRBO': {'topk': 10},
    'WordEmbeddingsInvertedRBOCentroid': {'topk': 10},
    'DocPerplexity': {},
    'WordPerplexity': {}
}


def evaluate_metric(eval_type, model_output, evaluation_dataset, params=None):
    if params is None:
        params = {}
    evaluation = None

    eval_params = Evaluation_params.get(eval_type, {})
    eval_params = set_params(eval_params, params)
    if eval_type in ['F1', 'Precision', 'Recall', 'Accuracy']:
        evaluation = Evaluations[eval_type](evaluation_dataset, **eval_params)
    elif eval_type in ['NPMI', 'C_V', 'U_MASS', 'C_UCI']:
        evaluation = Evaluations[eval_type](evaluation_dataset.get_corpus(), **eval_params)
    elif eval_type in ['WECoherencePairwise', 'WECoherenceCentroid', 'TopicDiversity', 'InvertedRBO',
                       'WordEmbeddingsInvertedRBO', 'WordEmbeddingsInvertedRBOCentroid',
                       'DocPerplexity', 'WordPerplexity']:
        evaluation = Evaluations[eval_type](**eval_params)

    if evaluation is None:
        raise ValueError('Evaluation metric is not defined')
    score = evaluation.score(model_output)
    return score


def evaluate_output(eval_type, output_dir, evaluation_dataset, params=None):
    if params is None:
        params = {}
    evaluation = None

    model_output = {}
    eval_params = Evaluation_params.get(eval_type, {})
    eval_params = set_params(eval_params, params)
    if eval_type in ['F1', 'Precision', 'Recall', 'Accuracy']:
        evaluation = Evaluations[eval_type](evaluation_dataset, **eval_params)
        model_output['topic-document-matrix'] = np.genfromtxt(os.path.join(output_dir, 'topic-document-matrix.txt'), dtype=None)
        model_output['test-topic-document-matrix'] = np.genfromtxt(os.path.join(output_dir, 'test-topic-document-matrix.txt'), dtype=None)

    elif eval_type in ['NPMI', 'C_V', 'U_MASS', 'C_UCI']:
        evaluation = Evaluations[eval_type](evaluation_dataset.get_corpus(), **eval_params)
        model_output['topics'] = np.genfromtxt(os.path.join(output_dir, 'topics.txt'), dtype=str)
    elif eval_type in ['WECoherencePairwise', 'WECoherenceCentroid', 'TopicDiversity', 'InvertedRBO',
                       'WordEmbeddingsInvertedRBO', 'WordEmbeddingsInvertedRBOCentroid']:
        evaluation = Evaluations[eval_type](**eval_params)
        model_output['topics'] = np.genfromtxt(os.path.join(output_dir, 'topics.txt'), dtype=str)

    elif eval_type == 'DocPerplexity':
        evaluation = Evaluations[eval_type](**eval_params)
        model_output['doc-losses'] = np.genfromtxt(os.path.join(output_dir, 'doc-losses.txt'), dtype=None)

    elif eval_type == 'WordPerplexity':
        evaluation = Evaluations[eval_type](**eval_params)
        model_output['word-losses'] = np.genfromtxt(os.path.join(output_dir, 'word-losses.txt'), dtype=None)

    if evaluation is None:
        raise ValueError('Evaluation metric is not defined')
    score = evaluation.score(model_output)
    return score


def set_params(params_1, params_2):
    for k in params_2.keys():
        if k in params_1.keys():
            params_1[k] = params_2.get(k, params_1[k])
    return params_1
