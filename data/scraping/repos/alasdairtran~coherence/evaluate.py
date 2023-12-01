import json
import logging
import os

from allennlp.common.util import prepare_environment
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate

from coherence.corpus import Corpus

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def evaluate_from_args(archive_path, overrides=None):
    """Evaluate on test data."""

    # Load from archive
    device = 0
    archive = load_archive(archive_path, device, overrides)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    corpus = Corpus.from_params(config.pop('corpus'))
    iterator_params = config.pop('validation_iterator', None)
    if not iterator_params:
        iterator_params = config.pop('iterator', None)
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)

    try:
        evaluate_dataset(corpus.valid, 'validation', model,
                         iterator, device, archive_path)

        evaluate_dataset(corpus.test, 'test', model,
                         iterator, device, archive_path)
    except (KeyboardInterrupt) as e:
        logger.warning(f'Evaluation is interrupted due to {e}. Exiting.')


def evaluate_dataset(instances, name, model, iterator, device, archive_path):
    logger.info(f'Evaluating {name} set.')
    metrics = evaluate(model, instances, iterator,
                       device, batch_weight_key='sample_size')

    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    output_dir = os.path.dirname(archive_path)
    output_file = os.path.join(output_dir, f'{name}-metrics.json')
    if output_file:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)
    return metrics
