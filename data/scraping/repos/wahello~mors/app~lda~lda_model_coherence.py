import logging
import sys

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaMulticore


from search_engine.configuration import LdaConfig
from search_engine.model import init_logger
from search_engine.preprocessing import Preprocessor
from search_engine.model import parse_dir_json

if __name__ == '__main__':
    init_logger()
    log = logging.getLogger('lda_model')

    config = LdaConfig(sys.argv[1]).get_current_config()
    config = Config(profile='local').lda

    _, docs = zip(*parse_dir_json(config['data_path']))

    preprocessed_docs = Preprocessor(max_workers=config['max_workers']).process_docs(docs)

    log.info("Loading model from %s", config['model_path'])
    lda_model = LdaMulticore.load(config['model_path'])
    log.info("Loading dictionary from %s", config['dict_path'])
    dictionary = Dictionary.load(config['dict_path'])

    coherence_model_lda = CoherenceModel(
        model=lda_model,
        texts=preprocessed_docs,
        dictionary=dictionary,
        coherence='c_v')

    coherence_lda = coherence_model_lda.get_coherence()

    import csv

    with open(config['coherence_path'], "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow([config['topics'], coherence_lda])
