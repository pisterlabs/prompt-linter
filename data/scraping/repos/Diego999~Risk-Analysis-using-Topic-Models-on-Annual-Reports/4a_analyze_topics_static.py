import os
import glob
import re
import stanford_corenlp_pywrapper
import utils
import logging
import sys
import numpy
import collections
import random
from multiprocessing import Process, Manager
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, HdpModel, LdaMulticore
from gensim.models.wrappers import LdaMallet
config = __import__('0_config')


try:
    import pyLDAvis.gensim
    CAN_VISUALIZE = True
except ImportError:
    CAN_VISUALIZE = False


def read_section(section):
    if not os.path.isdir(section):
        print('ERR: {} does not exists !'.format(section))
        exit(-1)

    return glob.glob(section + '/*.' + config.EXTENSION_10K_REPORT)


def remove_header_and_multiple_lines(buffer):
    text = ''.join(buffer[1:])
    text = re.sub(r'\n+', '\n', text).strip()
    return text.split('\n')


def content_relevant(buffer, KEYWORDS_TO_DETECT, MIN_LENGTH_EMPTY, MIN_LENGTH_KEYWORDS, MIN_LINES_KEYWORDS, REMOVE_START_WORDS):
    text = ''.join(buffer).lower()
    if len(text) < MIN_LENGTH_EMPTY:
        return False

    if len(buffer) < MIN_LINES_KEYWORDS and len(text) < MIN_LENGTH_KEYWORDS:
        for kw in KEYWORDS_TO_DETECT:
            if kw in text:
                return False

    return True


pattern_remove_multiple_space = re.compile(r'\s+')
pattern_remove_new_lines = re.compile(r'\n+')
pattern_non_char_digit_hash_perc_dash = re.compile(r'[^a-z0-9#%\' -]+')
pattern_remove_multiple_dash = re.compile(r'-+')
pattern_map_digit_to_hash = re.compile(r'[0-9]+')
pattern_html_tags = re.compile('<.*?>')
pattern_multiple_hash = re.compile(r'#+')
def clean(buffer, KEYWORDS_TO_DETECT, MIN_LENGTH_EMPTY, MIN_LENGTH_KEYWORDS, MIN_LINES_KEYWORDS, REMOVE_START_WORDS):
    text = ' '.join(buffer).lower()
    text = re.sub(pattern_html_tags, ' ', text)
    text = re.sub(pattern_non_char_digit_hash_perc_dash, ' ', text)
    text = re.sub(pattern_map_digit_to_hash, '#', text)
    text = re.sub(pattern_remove_multiple_dash, '-', text)
    text = re.sub(pattern_remove_new_lines, ' ', text)
    text = re.sub(pattern_remove_multiple_space, ' ', text)
    text = re.sub(pattern_multiple_hash, '#', text)
    text = text.strip()

    for remove_start_words in REMOVE_START_WORDS:
        if text.startswith(remove_start_words):
            text = text[len(remove_start_words) + 1:]
            while text[0] in [' ', '-', '#']:
                text = text[1:]

    return text.strip()


def load_and_clean_data_process(items, section, storage, pid):
    cleaned_data = []
    for i, item in enumerate(items):
        buffer = []
        with open(item, 'r', encoding='utf-8') as fp:
            for l in fp:
                buffer.append(l)

        buffer = remove_header_and_multiple_lines(buffer)
        if content_relevant(buffer, *config.CLEAN_PARAMETERS[section]):
            cleaned_data.append((item, clean(buffer, *config.CLEAN_PARAMETERS[section])))

    storage[pid] = cleaned_data


def load_and_clean_data(section):
    cleaned_data_file = section + config.SUFFIX_CLEAN_DATA
    cleaned_data = []
    if config.FORCE_PREPROCESSING or not os.path.exists(cleaned_data_file):
        items = read_section(section)
        manager = Manager()

        if config.MULTITHREADING:
            items_chunk = utils.chunks(items, 1 + int(len(items) / config.NUM_CORES))
            final_data = manager.list([[]] * config.NUM_CORES)

            procs = []
            for i in range(config.NUM_CORES):
                procs.append(Process(target=load_and_clean_data_process, args=(items_chunk[i], section, final_data, i)))
                procs[-1].start()

            for p in procs:
                p.join()
        else:
            final_data = [None]
            load_and_clean_data_process(items, section, final_data, 0)
        cleaned_data = sum(final_data, [])

        print('{}: Keep {} over {} ({:.2f}%)'.format(section[section.rfind('/') + 1:], len(cleaned_data), len(items), float(len(cleaned_data)) / len(items) * 100.0))

        with open(cleaned_data_file, 'w', encoding='utf-8') as fp:
            for i, l in cleaned_data:
                fp.write(i + '\t' + l + '\n')
    else:
        with open(cleaned_data_file, 'r', encoding='utf-8') as fp:
            for l in fp:
                i, l = l.split('\t')
                cleaned_data.append((i, l.strip()))

    return cleaned_data


def get_lemmas(parsed_data):
    assert len(parsed_data) == 1
    parsed_data = parsed_data[0]

    # Lemmatize
    lemmas = [l.lower() for l in parsed_data['lemmas']]

    # Lemmatize & remove proper nouns
    #assert len(parsed_data['lemmas']) == len(parsed_data['pos'])
    #lemmas = [l.lower() for l, p in zip(parsed_data['lemmas'], parsed_data['pos']) if 'NP' not in p]

    # Lemmatize & keep only nouns, adjectives and verbs
    #assert len(parsed_data['lemmas']) == len(parsed_data['pos'])
    #lemmas = [l.lower() for l, p in zip(parsed_data['lemmas'], parsed_data['pos']) if p in ['NN', 'NNS'] or 'VB' in p or 'JJ' in p]

    return lemmas


def get_tokens(parsed_data):
    assert len(parsed_data) == 1
    parsed_data = parsed_data[0]

    # Tokenize
    tokens = [l.lower() for l in parsed_data['tokens']]

    return tokens


def preprocess_util_thread(data, storage, pid):
    annotator = stanford_corenlp_pywrapper.CoreNLP(configdict={'annotators': 'tokenize, ssplit, pos, lemma'}, corenlp_jars=[config.SF_NLP_JARS])

    new_data = []
    for item, text in data:
        parsed_data = annotator.parse_doc(text)['sentences']
        lemmas = get_lemmas(parsed_data)
        new_data.append((item, lemmas))

    storage[pid] = new_data


def construct_lemma_dict_and_transform_lemma_to_idx(lemmas, lemma_to_idx, idx):
    # Construct dictionary
    for lemma in lemmas:
        if lemma not in lemma_to_idx:
            lemma_to_idx[lemma] = idx
            idx += 1

    # Replace lemma by their corresponding indices
    lemmas_idx = [lemma_to_idx[lemma] for lemma in lemmas]

    return lemmas_idx, lemma_to_idx, idx


def add_bigrams_trigrams(data):
    docs = [x[1] for x in data]

    bigrams = Phrases(docs, min_count=config.MIN_FREQ_BI) # a b -> a_b
    trigrams = Phrases([bigrams[docs[x]] for x in range(len(docs))], min_count=config.MIN_FREQ_TRI) # a b c -> a_b_c
    quadrigrams = Phrases([trigrams[docs[x]] for x in range(len(docs))], min_count=config.MIN_FREQ_QUA) # a b c d -> a_b_c_d

    for idx in range(len(data)):
        data[idx] = (data[idx][0], quadrigrams[docs[idx]])

    return data


def remove_stopwords(data):
    stopwords = set()
    with open(config.STOPWORD_LIST, 'r', encoding='utf-8') as fp:
        for l in fp:
            stopwords.add(l.strip().lower())

    # Remove stopwords
    new_data = []
    for item, lemmas in data:
        new_data.append((item, [l for l in lemmas if l not in stopwords and (len(l) > 1 and l != '#')]))

    return new_data


def transform_bow(data):
    new_data = []
    idx = 1
    lemma_to_idx = {'PAD': 0}
    for item, lemmas in data:
        lemmas_idx, lemma_to_idx, idx = construct_lemma_dict_and_transform_lemma_to_idx(lemmas, lemma_to_idx, idx)
        new_data.append((item, lemmas_idx))
    idx_to_lemma = {v: k for k, v in lemma_to_idx.items()}

    return new_data, lemma_to_idx, idx_to_lemma


def preprocess_util(data):
    manager = Manager()
    if config.MULTITHREADING:
        data = utils.chunks(data, 1 + int(len(data) / config.NUM_CORES))
        final_data = manager.list([[]] * config.NUM_CORES)

        procs = []
        for i in range(config.NUM_CORES):
            procs.append(Process(target=preprocess_util_thread, args=(data[i], final_data, i)))
            procs[-1].start()

        for p in procs:
            p.join()
    else:
        final_data = [None]
        preprocess_util_thread(data, final_data, 0)
    new_data = sum(final_data, [])

    new_data = add_bigrams_trigrams(new_data)
    new_data = remove_stopwords(new_data)
    new_data, lemma_to_idx, idx_to_lemma = transform_bow(new_data)

    return new_data, lemma_to_idx, idx_to_lemma


def preprocess(section, data):
    preprocessed_data_file = section + config.SUFFIX_PREPROCESSED_DATA
    dict_lemma_idx_file = section + config.DICT_LEMMA_IDX
    dict_idx_lemma_file = section + config.DICT_IDX_LEMMA

    preprocessed_data = None
    lemma_to_idx = None
    idx_to_lemma = None
    if config.FORCE_PREPROCESSING or not os.path.exists(preprocessed_data_file) or not os.path.exists(dict_lemma_idx_file) or not os.path.exists(dict_idx_lemma_file):
        preprocessed_data, lemma_to_idx, idx_to_lemma = preprocess_util(data)

        # Save files
        utils.save_pickle(preprocessed_data, preprocessed_data_file)
        utils.save_pickle(lemma_to_idx, dict_lemma_idx_file)
        utils.save_pickle(idx_to_lemma, dict_idx_lemma_file)
    else:
        # Load files
        preprocessed_data = utils.load_pickle(preprocessed_data_file)
        lemma_to_idx = utils.load_pickle(dict_lemma_idx_file)
        idx_to_lemma = utils.load_pickle(dict_idx_lemma_file)

    return (preprocessed_data, lemma_to_idx, idx_to_lemma)


def sort_by_year_annual_report(item):
    return utils.year_annual_report_comparator(utils.extract_year_from_filename_annual_report(item))


def preprocessing_topic(data, idx_to_lemma):
    for i in range(0, len(data)):
        data[i] = (data[i][0], [idx_to_lemma[idx] for idx in data[i][1]])

    # Sort data & create time slices to use dynamic topic modeling and analyze evolution of topics during the time (93 94 ... 2015 2016 2017 ...)
    data = sorted(data, key=lambda x: sort_by_year_annual_report(x[0]))
    data_years = [utils.extract_year_from_filename_annual_report(item) for item, text in data]
    time_slices = [freq for year, freq in sorted(collections.Counter(data_years).items(), key=lambda x: utils.year_annual_report_comparator(x[0]))]
    assert sum(time_slices) == len(data)

    texts = [x[1] for x in data]
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=config.MIN_FREQ, no_above=config.MAX_DOC_RATIO)
    corpus = [dictionary.doc2bow(x[1]) for x in data]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    return corpus, dictionary, texts, time_slices, data


# w.r.t. https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
def visualize(model, corpus, dictionary, html_filepath):
    prepared = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(prepared, html_filepath)
    if CAN_VISUALIZE:
        pyLDAvis.show(prepared)


# Chunksize should be fine w.r.t. https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation.pdf
def train_topic_model_or_load(corpus, dictionary, texts, num_topics=15, chunksize=2000, decay=0.5, offset=1.0, passes=10, iterations=400, eval_every=10, alpha='symmetric', eta='auto', model_file=None, only_viz=False):
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    if model_file is None:
        # LDA https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation.pdf
        #model = train_lda_model(corpus, chunksize, eval_every, id2word, iterations, num_topics, passes, decay, offset, alpha, eta)
        # LDA Multicore
        model = train_lda_model_multicores(corpus, chunksize, eval_every, id2word, iterations, num_topics, passes, decay, offset, alpha, eta, workers=config.NUM_CORES)
        # HDP http://proceedings.mlr.press/v15/wang11a/wang11a.pdf
        #model = train_hdp_model(corpus, dictionary, chunksize); only_viz = True
    else:
        # LDA
        # model = load_lda_model(model_file)
        # LDA Multicore
        model = load_lda_model_multicores(model_file)
        # HDP
        #model = load_hdp_model(model_file); only_viz = True

    c_v = compute_c_v(model, texts, dictionary, processes=config.NUM_CORES) if not only_viz else 0
    u_mass = compute_u_mass(model, texts, dictionary, processes=config.NUM_CORES) if not only_viz else 0

    return model, c_v, u_mass


# Cannot be visualized
def train_lda_mallet_model(corpus, eval_every, id2word, iterations, num_topics, workers=config.NUM_CORES):
    model = LdaMallet('./Mallet/bin/mallet', corpus=corpus, id2word=id2word, iterations=iterations, num_topics=num_topics, optimize_interval=eval_every, workers=workers, prefix='/tmp/')
    return model


def load_lda_mallet_model(filepath):
    return LdaMallet.load(filepath)


# https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation.pdf
def train_lda_model(corpus, chunksize, eval_every, id2word, iterations, num_topics, passes, decay, offset, alpha='auto', eta='auto'):
    print('LDA model')
    model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, alpha=alpha, eta=eta, decay=decay, offset=offset, iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every, random_state=config.SEED)
    return model


def load_lda_model(filepath):
    return LdaModel.load(filepath)


def train_lda_model_multicores(corpus, chunksize, eval_every, id2word, iterations, num_topics, passes, decay, offset, alpha='symmetric', eta='auto', workers=int(config.NUM_CORES)/2 - 1):
    print('LDA Multicore model')
    model = LdaMulticore(corpus=corpus, id2word=id2word, chunksize=chunksize, alpha=alpha, eta=eta, iterations=iterations, decay=decay, offset=offset, num_topics=num_topics, passes=passes, eval_every=eval_every, workers=workers, random_state=config.SEED)
    return model


def load_lda_model_multicores(filepath):
    return LdaMulticore.load(filepath)


# http://proceedings.mlr.press/v15/wang11a/wang11a.pdf
# Does not support coherence
def train_hdp_model(corpus, dictionary, chunksize):
    print('HDP model')
    model = HdpModel(corpus=corpus, id2word=dictionary, chunksize=chunksize, random_state=config.SEED)
    # To get the topic words from the model
    topics = []
    for topic_id, topic in model.show_topics(num_topics=10, formatted=False):
        topic = [word for word, _ in topic]
        topics.append(topic)
    return model


def load_hdp_model(filepath):
    return HdpModel.load(filepath)


# C_V should be the best estimator w.r.t http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf
def compute_c_v(model, texts, dictionary, processes=config.NUM_CORES):
    return CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v', processes=processes).get_coherence()


def compute_u_mass(model, texts, dictionary, processes=config.NUM_CORES):
    return CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass', processes=processes).get_coherence()


def tune_topic_model_process(corpus, dictionary, texts, num_topics_range, chunksize, decay, offset, alpha='symmetric', eta='auto', passes=10, iterations=400, eval_every=10):
    for num_topics in num_topics_range:
        train_and_write_score_topic_model(corpus, dictionary, texts, num_topics, chunksize, decay, offset, alpha, eta, passes, iterations, eval_every)


def train_and_write_score_topic_model(corpus, dictionary, texts, num_topics, chunksize, decay, offset, alpha='symmetric', eta='auto', passes=10, iterations=400, eval_every=10):
    print(num_topics, chunksize, decay, offset, alpha, eta, passes, iterations, eval_every)
    model, c_v, u_mass = train_topic_model_or_load(corpus, dictionary, texts, num_topics=num_topics, alpha=alpha, eta=eta, chunksize=chunksize, decay=decay, offset=offset, passes=passes, iterations=iterations, eval_every=eval_every, only_viz=False)
    filename = section[section.rfind('/') + 1:] + '_k:' + str(decay) + '_tau:' + str(offset) + '_alpha:' + str(alpha) + '_eta:' + str(eta) + '_topics:' + str(num_topics) + '_cu:' + str(round(u_mass, 4)) + '_cv:' + str(round(c_v, 4)) + '_rnd:' + str(config.SEED) + '.txt'
    print(filename)
    filename = os.path.join(config.OUTPUT_FOLDER, filename)
    with open(filename, 'w') as fp:
        fp.write(filename + '\n')


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.INFO)
    numpy.random.seed(config.SEED)
    random.seed(config.SEED)

    if not os.path.exists(config.OUTPUT_FOLDER):
        os.makedirs(config.OUTPUT_FOLDER)
    if not os.path.exists(config.MODEL_FOLDER):
        os.makedirs(config.MODEL_FOLDER)

    sections_to_analyze = [config.DATA_1A_FOLDER]
    for section in sections_to_analyze:
        data = load_and_clean_data(section)
        data, lemma_to_idx, idx_to_lemma = preprocess(section, data)
        corpus, dictionary, texts, time_slices, _ = preprocessing_topic(data, idx_to_lemma)

        # Train
        if not config.TUNING:
            model_filepath = config.TRAIN_PARAMETERS[section][1]
            model_file = model_filepath if os.path.exists(model_filepath) else None
            num_topics = config.TRAIN_PARAMETERS[section][0]

            params = config.TRAIN_PARAMETERS[section][4]
            if params is not None:
                model, c_v, u_mass = train_topic_model_or_load(corpus, dictionary, texts, model_file=model_file, only_viz=config.DO_NOT_COMPUTE_COHERENCE, **params)
            else: # Default
                model, c_v, u_mass = train_topic_model_or_load(corpus, dictionary, texts, num_topics=num_topics, alpha='symmetric', eta='auto', chunksize=2000, decay=0.5, offset=1.0, passes=10, iterations=400, eval_every=10, model_file=model_file, only_viz=False)#config.DO_NOT_COMPUTE_COHERENCE)
            if model_file is None:
                model.save(model_filepath)
            print('c_v:' + str(round(c_v, 4)) + ', cu:' + str(round(u_mass, 4)))
            visualize(model, corpus, dictionary, config.TRAIN_PARAMETERS[section][2])
        else: # Tune
            assert len(sys.argv) == 4 or len(sys.argv) == 7
            only_topics, max_try, seed = [int(x) for x in sys.argv[1:4]]
            chunksize, kappa, tau = [float(x) for x in sys.argv[4:]] if len(sys.argv) > 4 else [2000, 0.5, 1.0]
            chunksize, tau = int(chunksize), int(tau)
            numpy.random.seed(seed)
            random.seed(seed)
            print('Args: ', only_topics, max_try, seed, chunksize, kappa, tau)

            # Tune only number of topics
            if only_topics == -1:
                nb_parallel_runs = max_try
                topic_range = list(range(1, 101))
                random.shuffle(topic_range)
                topic_range = utils.chunks(topic_range, int(len(topic_range) / nb_parallel_runs))

                procs = []
                for i in range(nb_parallel_runs):
                    procs.append(Process(target=tune_topic_model_process, args=(corpus, dictionary, texts, topic_range[i], chunksize, kappa, tau, 'symmetric', 'auto', 10, 400, 10,)))
                    procs[-1].start()

                for p in procs:
                    p.join()
            else:
                alpha_range = ['asymmetric', 'symmetric']
                eta_range = ['symmetric', 'auto']
                topic_range = [6, 28]

                chuncksize_range = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
                kappa_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Decay
                tau_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]  # Offset

                for i in range(max_try):
                    alpha = utils.draw_val(alpha_range)
                    eta = utils.draw_val(eta_range)
                    num_topics = utils.draw_val(topic_range)
                    chunksize = utils.draw_val(chuncksize_range)
                    kappa = utils.draw_val(kappa_range)
                    tau = utils.draw_val(tau_range)

                    tune_topic_model_process(corpus, dictionary, texts, [num_topics], chunksize, kappa, tau, alpha, eta, passes=10, iterations=400, eval_every=10)
