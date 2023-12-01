import re
import csv, sys
from scipy.stats import ranksums
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from random import shuffle

import math
import spacy
import numpy as np
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
from gensim.models.wrappers import LdaMallet
from gensim.models import CoherenceModel
from pprint import pprint

sys.path.append('../')
from utils import Serialization


class Utils:
    @staticmethod
    def true_case(text, frequencies):
        """
        covert words' case into the most probable one based on corpus counts
        simple unigram-based procedure wherea word is assigned its most probable case based on
        the # of occurrences in the english wikipedia dataset
        :param text: text for true casing
        :param frequencies: (wikipedia-based) dictionary of case-sensitive word and their counts
        :return: true-cased text
        """
        tc_tokens = []
        for token in word_tokenize(text.strip()):
            lfreq = frequencies.get(token.lower(), 0)
            ufreq = frequencies.get(token.upper(), 0)
            cfreq = frequencies.get(token.capitalize(), 0)
            fmax = max([lfreq, ufreq, cfreq])
            if fmax < 200:
                tc_tokens.append(token)
                continue
            # end if
            if fmax == lfreq:
                tc_tokens.append(token.lower())
                continue
            # end if
            if fmax == ufreq:
                tc_tokens.append(token.upper())
                continue
            # end if
            if fmax == cfreq:
                tc_tokens.append(token.capitalize())
                continue
            # end if
        # end for
        return ' '.join(tc_tokens)

    # end def

    @staticmethod
    def remove_multiple_spaces(filename):
        """
        formatting textual data by removing redundant whitespaces
        :param filename: file to process
        """
        with open(filename, 'r') as fin, open(filename.replace('.csv', '_clean.csv'), 'w') as fout:
            csv_reader = csv.reader(fin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = csv_reader.__next__()
            csv_writer.writerow(header[1:])
            for line in csv_reader:
                if len(line) < 7: continue
                line[8] = re.sub(r'\n\s*\n', '\n\n', line[8].strip())
                line[8] = re.sub('  ', ' ', line[8])
                csv_writer.writerow(line[1:])
            # end for
        # end with
    # end def

    @staticmethod
    def substitute_named_entities(filename, common_users):
        """
        true-case text and substitute named entities with their type (e.g., organization, person)
        true-casing precedes ner since it's case-sensitive
        :param filename: file for processing
        :param common_users: the set of user common to code-switched and monolingual text
        """
        object_name = '<frequencies dictionary object>'
        frequencies = Serialization.load_obj(object_name)
        nlp = spacy.load('en_core_web_lg', disable=['tokenizer', 'parser', 'tagger'])
        with open(filename, 'r') as fin, open(filename.replace('.csv', '_tc_ne.csv'), 'w') as fout:
            csv_reader = csv.reader(fin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(csv_reader.__next__())
            for line in csv_reader:
                if len(line) < 8: continue
                if line[0].strip() not in common_users: continue
                if len(line[7].split()) < 30: continue

                text_tc = Utils.true_case(line[7], frequencies)

                prev_end = 0
                line_with_entities = []
                for ent in nlp(text_tc).ents:
                    line_with_entities.append(''.join(text_tc[prev_end:ent.start_char]))
                    line_with_entities.append(ent.label_)
                    prev_end = ent.end_char
                # end for
                line_with_entities.append(''.join(text_tc[prev_end:]))
                line[7] = (' '.join(line_with_entities)).strip()
                csv_writer.writerow(line)

            # end for
        # end with
    # end def

    @staticmethod
    def post_to_words(data):
        """
        returns a list of text words
        :param data: text
        :return: a list fo words (punctuation excluded)
        """
        outdata = []
        for post in data:
            # deacc=True to remove punctuation
            #outdata.append(simple_preprocess(str(post), deacc=True))
            outdata.append(re.sub('[,.!?]', ' ', str(post)).split())
        # end for
        return outdata
    # end def

    @staticmethod
    def remove_noncontent_words(data, stop_words, ranks):
        """
        given a set of posts, filter in only content words
        :param data: a list of posts (documents) fpr processing
        :param stop_words: a list of english function words
        :param ranks: a map of word to frequency rank
        :return: a list of posts with content words
        """
        docs = []
        for doc in data:
            words = []
            for word in doc:
                if len(word) not in range(4, 15): continue
                if word in NAMED_ENTITIES or word in stop_words: continue
                if ranks.get(word, 0) < MIN_WORD_RANK or ranks.get(word, sys.maxsize) > MAX_WORD_RANK: continue

                words.append(word.lower())
            # end for
            if len(words) < 10: continue
            docs.append(words)
        # end for
        return docs
    # end def

    @staticmethod
    def lemmatization(data):
        """
        https://spacy.io/api/annotation
        lemmatize text and filter-in POS tags meaningful for topic analysis
        :param data: text for processing
        :return: lemmatized anf filtered text
        """
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
        nlp = spacy.load('en_core_web_lg', disable=['tokenizer', 'parser', 'ner'])
        outdata = []
        for post in data:
            outpost = []
            doc = nlp(' '.join(post))
            for token in doc:
                if token.text in NAMED_ENTITIES: continue
                if token.pos_ in allowed_postags: outpost.append(token.lemma_)
            # end for
            outdata.append(outpost)
        # end for
        return outdata
    # end def

    @staticmethod
    def get_wikipedia_word_ranked_list():
        """
        create and save two dictionaries: word to rank, and word to count
        """
        wordcount = {}
        filename = '<english wikipedia dump location>'
        with open(filename, 'r') as fin:
            for line in fin:
                for token in line.split():
                    count = wordcount.get(token, 0)
                    wordcount[token] = count + 1
                # end for
            # end for
            sorted_wordcount = sorted(wordcount, key=wordcount.get, reverse=True)

            ranks = {}
            for count, key in enumerate(sorted_wordcount):
                if count > 500000: continue
                ranks[key] = count
            # end for
        # end with
        Serialization.save_obj(wordcount, 'dict.counts.cs')
        Serialization.save_obj(ranks, 'dict.ranks.cs')
    # end def

    @staticmethod
    def extract_users_common_set():
        """
        extract a set of user with both code-switched and english monolingual posts
        """
        users_cs = []
        filename = '<a csv file with code-switched posts>'
        with open(filename, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = csv_reader.__next__()
            for line in csv_reader:
                if len(line) < 8: continue
                users_cs.append(line[0].strip())
            # end for
        # end with

        users_non_cs = []
        filename = '<a csv file with monolingual enlgish posts>'
        with open(filename, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = csv_reader.__next__()
            for line in csv_reader:
                if len(line) < 8: continue
                users_non_cs.append(line[0].strip())
            # end for
        # end with

        common_users = set(users_cs).intersection(set(users_non_cs))
        print('total cs users, monolingual users, common users:', len(set(users_cs)),
              len(set(users_non_cs)), len(common_users))

        Serialization.save_obj(common_users, 'common.users')

    # end def

    @staticmethod
    def lemmatization_and_pos_filter(filename, common_users):
        """
        preprocessing data towards topic modeling
        :param filename: a csv file with code-switched or monolingual data
        :param common_users: a list of user with both types of posts
        """
        stop_words = stopwords.words('english')
        with open(filename, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = csv_reader.__next__()

            data = []
            for line in csv_reader:
                if len(line) < 8: continue
                if len(line[7].split()) < MIN_SENTENCE_LENGTH: continue
                if line[0].strip() not in common_users: continue
                data.append(line[7])
            # end for
        # end with

        print('total of', len(data), 'posts')
        tokens = sum([len(post.split()) for post in data])
        print('average post length', float(tokens)/len(data))

        print('converting posts to words...')
        data_words = list(Utils.post_to_words(data))
        print('skipping (performing) lemmatization and pos filtering...')
        data_words = Utils.lemmatization(data_words)
        print('removing stopwords and unfrequent words...')
        ranks = Serialization.load_obj('dict.ranks')
        data_words = Utils.remove_noncontent_words(data_words, stop_words, ranks)

        Serialization.save_obj(data_words, current_mode+'.preprocessed')

    # end def

    @staticmethod
    def topic_modelling(data_object_name):
        """
        perform topic modelign for a given set of posts (data object)
        :param data_object_name: raw data for topic modeling
        """
        data_words = Serialization.load_obj(data_object_name)

        stop_words = stopwords.words('english')
        print('removing stopwords and unfrequent words...')
        ranks = Serialization.load_obj('dict.ranks')
        data_words = Utils.remove_noncontent_words(data_words, stop_words, ranks)

        id2word = corpora.Dictionary(data_words)
        corpus = [id2word.doc2bow(post) for post in data_words]

        topics = CS_TOPICS
        print('performing topic modeling with', topics, 'topics')
        ldamodel = LdaMallet(mallet_path, corpus=corpus, num_topics=topics, id2word=id2word)
        pprint(malletmodel2ldamodel(ldamodel).top_topics(corpus, data_words, id2word))

        '''
        pprint(ldamodel.show_topics(num_topics=min([20, topics]), num_words=20, formatted=False))
        coherence_model = CoherenceModel(model=ldamodel, texts=data_words, dictionary=id2word, coherence='c_v')
        coherence = coherence_model.get_coherence()
        print('\ntopics:\t', topics, 'coherence score:\t', coherence)
        sys.stdout.flush()
        '''

    # end def

    @staticmethod
    def topical_differences_sig_analysis():
        """
        testing code-switching and monolingual english posts for topical differences
        (1) partition code-switched posts into two random sets
        (2) perform topic modeling of each partition and compute the similarity between the two parts and
        their individual similarity to topics extracted from monolingual posts
        (3) test the multiple-experiment similarity scores for significance
        """
        data_object_name = 'monolingual.preprocessed'

        data_words = Serialization.load_obj(data_object_name)

        stop_words = stopwords.words('english')
        print('removing stopwords and infrequent words...')
        ranks = Serialization.load_obj('dict.ranks')
        data_words = Utils.remove_noncontent_words(data_words, stop_words, ranks)
        print('after pre-processing: total of', len(data_words), 'posts')

        topics = MONOLINGUAL_TOPICS
        for i in range(EXPERIMENTS):
            shuffle(data_words)
            part1 = data_words[:math.floor(len(data_words)/2)]
            part2 = data_words[math.floor(len(data_words)/2):]

            model = Utils.model_topic(part1, topics)
            Serialization.save_obj(model, 'lda.mallet.monolingual.part1.'+str(i))
            print('saved topic model: part1,', i)

            model = Utils.model_topic(part2, topics)
            Serialization.save_obj(model, 'lda.mallet.monolingual.part2.'+str(i))
            print('saved topic model: part2,', i)
            sys.stdout.flush()

        # end for

        inter = []; intra = []
        ldamodel_cs = malletmodel2ldamodel(Serialization.load_obj('lda.mallet.cs'))
        for i in range(30):
            print('processing', i)
            ldamodel_mono1 = malletmodel2ldamodel(Serialization.load_obj('lda.mallet.monolingual.part1.'+str(i)))
            ldamodel_mono2 = malletmodel2ldamodel(Serialization.load_obj('lda.mallet.monolingual.part2.'+str(i)))
            diff_matrix1, _ = ldamodel_cs.diff(ldamodel_mono1, distance='jaccard')
            diff_matrix2, _ = ldamodel_cs.diff(ldamodel_mono2, distance='jaccard')
            #intra.append(np.mean([np.mean(np.matrix(diff_matrix1)), np.mean(np.matrix(diff_matrix2))]))
            intra.append(np.mean([np.min(np.matrix(diff_matrix1)), np.min(np.matrix(diff_matrix2))]))
            diff_matrix3, _ = ldamodel_mono1.diff(ldamodel_mono2, distance='jaccard')
            #inter.append(np.mean(np.matrix(diff_matrix3)))
            inter.append(np.min(np.matrix(diff_matrix3)))
        # end for

        print(np.mean(intra), np.mean(inter))
        _, pval = ranksums(intra, inter)
        print('pval:', pval)

    # end def

    @staticmethod
    def model_topic(data_words, topics):
        """
        return topics model given data and number of topics
        :param data_words: data for topic modeling (e.g., a set of posts)
        :param topics: number of desired topics
        :return: topic model
        """
        id2word = corpora.Dictionary(data_words)
        corpus = [id2word.doc2bow(post) for post in data_words]
        print('performing topic modeling with', topics, 'topics')
        return LdaMallet(mallet_path, corpus=corpus, num_topics=topics, id2word=id2word)
    # end def

# end class


class DataProcessing:
    """
    a set of auxiliary methods for data processing
    """
    @staticmethod
    def clean_and_prepare_data():
        filename = '<filename for cleaning>'
        Utils.remove_multiple_spaces(filename)
        Utils.get_wikipedia_word_ranked_list()
        Utils.extract_users_common_set()

    # end def

    @staticmethod
    def test_true_casing():
        frequencies = Serialization.load_obj('dict.counts.cs')
        text = 'what do you think about john? i believe he is from toronto!'
        tc = Utils.true_case(text, frequencies)
        print(tc)

    # end def

# end class


EXPERIMENTS = 30
CS_TOPICS = 17
MONOLINGUAL_TOPICS = 21
MIN_WORD_RANK = 300
MAX_WORD_RANK = 10000
MIN_SENTENCE_LENGTH = 50
NAMED_ENTITIES = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT',
                  'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT',
                  'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

mallet_path = '<path-to-mallet-topic-modeling-dir>mallet-2.0.8/bin/mallet'
current_mode = 'monolingual'  # cs

if __name__ == '__main__':

    DataProcessing.test_true_casing()
    DataProcessing.clean_and_prepare_data()
    common_users = Serialization.load_obj('common.users')

    filename = 'data/'+current_mode+'_corpus_clean.csv'
    Utils.substitute_named_entities(filename, common_users)
    filename = 'data/'+current_mode+'_corpus_clean_tc_ne.csv'
    Utils.lemmatization_and_pos_filter(filename, common_users)

    Utils.topical_differences_sig_analysis()

# end if
