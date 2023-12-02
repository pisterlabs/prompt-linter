"""Train and run language models.

This program trains and evaluates language models using reinforcement learning.

Usage:
    coherence train [options] PARAM_PATH
    coherence evaluate [options] ARCHIVE_PATH
    coherence (-h | --help)
    coherence (-v | --version)

Options:
    -e --expt-dir EXPT_PATH
                        Directory to store experiment results and model files.
                        If not given, they will be stored in the same directory
                        as the parameter file.

    -r, --recover       Recover training from existing model.

    -f, --del-models    Delete existing models and logs.

    -b, --del-vocab     Delete existing vocab.

    -o --overrides OVERRIDES
                        A JSON structure used to override the experiment
                        configuration.

    -u --pudb           Enable debug mode with pudb.

    -p --ptvsd PORT     Enable debug mode with ptvsd on a given port, for
                        example 5678.

    -g --file-friendly-logging
                        Outputs tqdm status on separate lines and slows tqdm
                        refresh rate

    -i --include-package PACKAGE
                        Additional packages to include.

    -c --convert        Turn on convert mode without training. This is useful
                        when loading pretrained weights from another framework,
                        for example fairseq.

    -q --quiet          Print less info

    PARAM_PATH          Path to file describing the model parameters.

    ARCHIVE_PATH        Path to the archive file.

Examples:
    python -m coherence.commands train experiments/entity_grid_ranking/config.yaml -fb
"""

import logging
import os

import ptvsd
import pudb
from docopt import docopt
from schema import And, Or, Schema, Use

from coherence.utils import setup_logger

from .evaluate import evaluate_from_args
from .train import train_model_from_file

logger = setup_logger()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'param_path': Or(None, os.path.exists),
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        object: object,
    })
    args = schema.validate(args)
    args['debug'] = args['ptvsd'] or args['pudb']
    return args


def main():
    """Parse command line arguments and execute script."""
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['debug']:
        logger.setLevel(logging.DEBUG)
    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address, redirect_output=True)
        ptvsd.wait_for_attach()
    elif args['pudb']:
        pudb.set_trace()

    if args['quiet']:
        # Disable some of the more verbose logging statements
        logging.getLogger('allennlp.common.params').disabled = True
        logging.getLogger('allennlp.nn.initializers').disabled = True
        logging.getLogger(
            'allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    if args['train']:
        train_model_from_file(args['param_path'], args['expt_dir'],
                              args['overrides'], args['file_friendly_logging'],
                              args['recover'], args['del_models'],
                              args['del_vocab'], args['convert'])

    elif args['evaluate']:
        evaluate_from_args(args['archive_path'], args['overrides'])


if __name__ == '__main__':
    main()
