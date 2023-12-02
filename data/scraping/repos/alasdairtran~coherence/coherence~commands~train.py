"""Train models using config files.

.. code-block:: bash

   $ allennlp train --help

   usage: allennlp train [-h] -s SERIALIZATION_DIR [-r] [-f] [-o OVERRIDES]
                         [--file-friendly-logging]
                         [--include-package INCLUDE_PACKAGE]
                         param_path

   Train the specified model on the specified dataset.

   positional arguments:
     param_path            path to parameter file describing the model to be
                           trained

   optional arguments:
     -h, --help            show this help message and exit
     -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                           directory in which to save the model and its logs
     -r, --recover         recover training from the state in serialization_dir
     -f, --force           overwrite the output directory if it exists
     -o OVERRIDES, --overrides OVERRIDES
                           a JSON structure used to override the experiment
                           configuration
     --file-friendly-logging
                           outputs tqdm status on separate lines and slows tqdm
                           refresh rate
     --include-package INCLUDE_PACKAGE
                            additional packages to include
"""

import itertools
import logging
import os
import re
import shutil
import sys
from glob import glob
from typing import Any, Dict, List, Tuple

import torch
from allennlp.common import Params
from allennlp.common.checks import check_for_gpu
from allennlp.common.params import Params
from allennlp.common.util import dump_metrics, prepare_global_logging
from allennlp.data import Vocabulary
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import CONFIG_NAME, archive_model
from allennlp.models.model import _DEFAULT_WEIGHTS, Model
from allennlp.nn import RegularizerApplicator
from allennlp.training.trainer import Trainer
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.util import create_serialization_dir, evaluate

from coherence.training import TrainerF16SingleTask, TrainerPieces
from coherence.utils import load_params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def train_model_from_file(parameter_filename: str,
                          serialization_dir: str,
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          recover: bool = False,
                          del_models: bool = False,
                          del_vocab: bool = False,
                          convert: bool = False) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    parameter_filename : ``str``
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`train_model`.
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    del_models : ``bool``, optional (default=False)
        If ``True``, we will delete existing models and logs if they already exist.
    del_vocab : ``bool``, optional (default=False)
        If ``True``, we will delete existing vocabulary if it already exists.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = load_params(parameter_filename, overrides)
    if not serialization_dir:
        serialization_dir = os.path.dirname(parameter_filename)
    return train_model(params, serialization_dir, file_friendly_logging,
                       recover, del_models, del_vocab, convert)


def train_model(params: Params,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                recover: bool = False,
                del_models: bool = False,
                del_vocab: bool = False,
                convert: bool = False) -> Model:
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    del_models : ``bool``, optional (default=False)
        If ``True``, we will delete existing models and logs if they already exist.
    del_vocab : ``bool``, optional (default=False)
        If ``True``, we will delete existing vocabulary if it already exists.

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    if cuda_device >= 0:
        check_for_gpu(cuda_device)
        torch.cuda.set_device(cuda_device)

    # Sometimes we might change the config a bit but still want to continue training
    # if recover:
    #     create_serialization_dir(
    #         params, serialization_dir, recover, del_models)
    if del_models:
        for path in glob(f'{serialization_dir}/*'):
            if os.path.isfile(path) and not path.endswith('config.yaml'):
                os.remove(path)
        log_path = f'{serialization_dir}/log'
        if os.path.isdir(log_path):
            shutil.rmtree(log_path)
    if del_vocab:
        vocab_path = f'{serialization_dir}/vocabulary'
        if os.path.isdir(vocab_path):
            shutil.rmtree(vocab_path)

    prepare_global_logging(serialization_dir, file_friendly_logging)

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    check_for_gpu(cuda_device)

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)

    trainer_type = params.get("trainer", {}).get("type", "default")

    if trainer_type == 'default':
        # Special logic to instantiate backward-compatible trainer.
        pieces = TrainerPieces.from_params(
            params, serialization_dir, recover)  # pylint: disable=no-member
        trainer = Trainer.from_params(
            model=pieces.model,
            serialization_dir=serialization_dir,
            iterator=pieces.iterator,
            train_data=pieces.corpus.train,
            validation_data=pieces.corpus.valid,
            params=pieces.params,
            validation_iterator=pieces.validation_iterator)
        evaluation_iterator = pieces.validation_iterator or pieces.iterator
        evaluation_dataset = pieces.corpus.test
        batch_weight_key = pieces.batch_weight_key

    elif trainer_type == 'trainer_fp16_single':
        params.get("trainer").pop('type')
        # Special logic to instantiate backward-compatible trainer.
        pieces = TrainerPieces.from_params(
            params, serialization_dir, recover)  # pylint: disable=no-member
        trainer = TrainerF16SingleTask.from_params(
            model=pieces.model,
            serialization_dir=serialization_dir,
            files_to_archive=params.files_to_archive,
            iterator=pieces.iterator,
            train_data=pieces.corpus.train,
            validation_data=pieces.corpus.valid,
            params=pieces.params,
            validation_iterator=pieces.validation_iterator)
        evaluation_iterator = pieces.validation_iterator or pieces.iterator
        evaluation_dataset = pieces.corpus.test
        batch_weight_key = pieces.batch_weight_key

    else:
        trainer = TrainerBase.from_params(params, serialization_dir, recover)
        # TODO(joelgrus): handle evaluation in the general case
        evaluation_iterator = evaluation_dataset = None

    params.assert_empty('base train command')

    if convert:
        logging.info('In conversion mode.')
        trainer._save_checkpoint(epoch=0)
        create_model_archive(serialization_dir, params)
        sys.exit(0)

    try:
        metrics = trainer.train()
    except (KeyboardInterrupt, RuntimeError):
        # if we have completed an epoch, try to create a model archive.
        logging.info("Training stopped. Attempting to create "
                     "a model archive using the current best epoch weights.")
        create_model_archive(serialization_dir, params)
        raise

    # Evaluate
    if evaluation_dataset and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(trainer.model, evaluation_dataset, evaluation_iterator,
                                cuda_device=trainer._cuda_devices[0],  # pylint: disable=protected-access,
                                # TODO(brendanr): Pass in an arg following Joel's trainer refactor.
                                batch_weight_key=batch_weight_key)

        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif evaluation_dataset:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)
    dump_metrics(os.path.join(serialization_dir,
                              "metrics.json"), metrics, log=True)

    # We count on the trainer to have the model with best weights
    return trainer.model


def create_model_archive(serialization_dir, params):
    if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
        archive_model(serialization_dir,
                      files_to_archive=params.files_to_archive)
    logger.info('Archiving is successful. Exiting.')
