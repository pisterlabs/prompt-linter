import pandas as pd
import numpy as np
import nltk
import string
from scipy.stats import entropy
import language_tool_python
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
from textstat.textstat import textstatistics
from textacy import preprocessing
import spacy
import collections
from lexrank import STOPWORDS, LexRank
from path import Path
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import gensim.corpora as corpora
from gensim.models import CoherenceModel


class DataEng:
    def __init__(self, input) -> None:
        # self.data = pd.DataFrame
        self.input = input
        self.words = nltk.word_tokenize(input)
        self.sent = nltk.sent_tokenize(input)
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

        self.documents = []
        self.documents_dir = Path('Text Document')

        for file_path in self.documents_dir.files('*.txt'):
            with file_path.open(mode='rt', encoding='utf-8') as fp:
                self.documents.append(fp.readlines())

    def Engineering(self):
        num_of_words = len(self.input.split())
        stopwords_freq = self.stopword_count() / num_of_words
        av_word_per_sen = num_of_words / len(self.sent)
        punctuations = self.punctuation()
        ARI = self.ARI(num_of_words)

        # Frequency of diff words
        tagging = self.POS_Tagging()
        freq_of_verb = tagging[0] / num_of_words
        freq_of_adj = tagging[1] / num_of_words
        freq_of_adv = tagging[2] / num_of_words
        freq_of_distinct_adj = tagging[6] / num_of_words
        freq_of_distinct_adv = tagging[7] / num_of_words
        freq_of_wrong_words = self.wrongwords()
        freq_of_noun = tagging[5] / num_of_words
        freq_of_transition = tagging[4] / num_of_words
        freq_of_pronoun = tagging[3] / num_of_words
        noun_to_adj = freq_of_adj / freq_of_noun
        verb_to_adv = freq_of_distinct_adv / freq_of_verb
        phrase_diversity = noun_to_adj * verb_to_adv

        # Sentiments Score
        sentiments = self.sentiment_Score()
        sentiment_compound = sentiments[0]
        sentiment_positive = sentiments[1]
        sentiment_negative = sentiments[2]

        # Grammar
        num_of_grammar_errors = self.grammarerrors()
        corrected_text = self.fixgrammar()
        num_of_short_forms = self.shortforms(corrected_text)
        Incorrect_form_ratio = (
            num_of_grammar_errors + num_of_short_forms) / num_of_words

        # Readability
        flesch_reading_ease = textstat.flesch_reading_ease(
            preprocessing.normalize.whitespace(self.input))
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(
            preprocessing.normalize.whitespace(self.input))
        dale_chall_readability_score = textstat.dale_chall_readability_score(
            preprocessing.normalize.whitespace(self.input))
        text_standard = textstat.text_standard(
            preprocessing.normalize.whitespace(self.input), float_output=True)
        # mcalpine_eflaw = 3.3
        mcalpine_eflaw = 25 - \
            textstat.mcalpine_eflaw(
                preprocessing.normalize.whitespace(self.input))
        # module 'textstat' has no attribute 'mcalpine_eflaw'

        number_of_diff_words = self.difficult_words()
        freq_diff_words = number_of_diff_words / num_of_words
        ttr = self.ttrscore(corrected_text)
        coherence_score = self.coherence_score()  # hv't edit
        lexrank_avg_min_diff, lexrank_interquartile = self.lexrank()
        sentence_complexity = self.sentence_complexity()

        output = [[num_of_words, stopwords_freq, av_word_per_sen, punctuations, ARI,
                   freq_of_verb, freq_of_adj, freq_of_adv, freq_of_distinct_adj, freq_of_distinct_adv,
                   sentence_complexity, freq_of_wrong_words, sentiment_compound, sentiment_positive,
                   sentiment_negative, num_of_grammar_errors,
                   num_of_short_forms, Incorrect_form_ratio, flesch_reading_ease, flesch_kincaid_grade,
                   dale_chall_readability_score, text_standard, mcalpine_eflaw, number_of_diff_words,
                   freq_diff_words, ttr, coherence_score,
                   lexrank_avg_min_diff, lexrank_interquartile, freq_of_noun, freq_of_transition, freq_of_pronoun,
                   noun_to_adj, verb_to_adv, phrase_diversity]]

        return pd.DataFrame(output)

# Helper functions
    # TTR
    def ttrscore(self, sample):
        sample_words = sample.split()
        n_words = len(sample_words)

        # remove all punctuations
        for i in range(n_words):
            for c in string.punctuation:
                sample_words[i] = sample_words[i].replace(c, '')

        # remove empty words
        sample_words = list(filter(None, sample_words))
        n_words = len(sample)

        # count each word
        word_count = collections.Counter(sample_words)

        # get the sorted list of unique words
        unique_words = list(word_count.keys())
        unique_words.sort()

        ttr = len(word_count)/float(n_words)
        return ttr

    # Difficult Word Extraction
    def syllables_count(self, word):
        return textstatistics().syllable_count(word)

    def break_sentences(self, text):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        return list(doc.sents)

    def difficult_words(self):  # python -m spacy download en
        nlp = self.nlp
        words = self.words
        # doc = nlp(self.input)
        # Find all words in the text
        diff_words_set = set()

        for word in words:
            syllable_count = self.syllables_count(word)
            if word not in nlp.Defaults.stop_words and syllable_count >= 2:
                diff_words_set.add(word)

        return len(diff_words_set)

    def shortforms(self, correct_text):
        collections = {'u': 'you', 'b': 'B'}
        num_of_error = 0
        splitted_text = correct_text.split()
        for word in splitted_text:
            if word in collections.keys():
                num_of_error += 1
        return num_of_error

    def fixgrammar(self):  # It returns the correct text
        tool = language_tool_python.LanguageTool('en-US')
        return tool.correct(self.input.replace('\n', ''))

    def grammarerrors(self):
        tool = language_tool_python.LanguageTool('en-US')
        return len(tool.check(self.input.replace('\n', '')))

    def wrongwords(self):
        spell = SpellChecker()
        wrong = spell.unknown(self.words)
        return len(wrong) / len(self.input.split())

    def punctuation(self):
        count = 0
        for word in self.words:
            count += len([c for c in word if c in list(string.punctuation)])
        return count

    def stopword_count(self):
        count = 0
        for word in self.words:
            if word in stopwords.words('english'):
                count += 1
        return count

    # Lexrank
    def lexrank(self):
        lxr = LexRank(self.documents, stopwords=STOPWORDS['en'])
        ranks = lxr.rank_sentences(self.sent, threshold=None)

        return sum(ranks)/len(ranks)-min(ranks), np.quantile(ranks, [0.25, 0.75])[1] - np.quantile(ranks, [0.25, 0.75])[1]

    def sentence_complexity(self):
        sentences = self.sent
        entropy_list = []
        for sentence in sentences:
            ps = PorterStemmer()
            stemmed = [ps.stem(word) for word in sentence]
            tags = nltk.pos_tag(stemmed)

            tag_dic = {}
            for tag in tags:
                if tag[1] in tag_dic.keys():
                    tag_dic[tag[1]] += 1
                else:
                    tag_dic[tag[1]] = 1

            entropy_list.append(entropy(list(tag_dic.values())))

        return sum(entropy_list) / len(entropy_list)

    def remove_stopwords(self, texts):
        stop_words = stopwords.words('english')
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(self, texts, bigram_mod):
        return [bigram_mod[doc] for doc in texts]

    def lemmatization(self, texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append(
                [token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def coherence_score(self):
        words = self.words
        bigram = gensim.models.Phrases(words, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        # Remove Stop Words
        data_words_nostops = self.remove_stopwords(words)
        # Form Bigrams
        data_words_bigrams = self.make_bigrams(data_words_nostops, bigram_mod)
        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(
            data_words_bigrams, self.nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)
        # Create Corpus
        texts = data_lemmatized
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # Build LDA model
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=5,
                                               random_state=100,
                                               chunksize=100,
                                               passes=10,
                                               per_word_topics=True)

        # Compute coherence score
        coherence_model_lda = CoherenceModel(
            model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        return coherence_lda

    def ARI(self, num_of_words):
        ttl_characters = 0
        for word in self.input.split():
            ttl_characters += len(word)
        ari = 4.71*(ttl_characters / num_of_words) + 0.5 * \
            (num_of_words / len(self.sent)) - 21.43
        return ari

    def sentiment_Score(self):
        # nltk.download('vader_lexicon') # Check if there is vader_lexicon -> if not download
        sia = SentimentIntensityAnalyzer()
        com = sia.polarity_scores(self.input).get('compound')
        pos = sia.polarity_scores(self.input).get('pos')
        neg = sia.polarity_scores(self.input).get('neg')
        return [com, pos, neg]

    def POS_Tagging(self):
        # Set the initial count num as 1 to avoid infinity in operation
        num_of_verb = 1
        num_of_adj = 1
        num_of_adv = 1
        num_of_pron = 1
        num_of_tran = 1
        num_of_noun = 1
        adj_list = []
        adv_list = []

        sample = self.words
        ps = PorterStemmer()
        sample_tokens = [ps.stem(word) for word in sample]

        tag = nltk.pos_tag(sample_tokens)
        for j in range(len(tag)):
            if tag[j][1] == 'VB':
                num_of_verb += 1
                continue
            if tag[j][1][:2] == 'JJ':
                num_of_adj += 1
                adj_list.append(tag[j][0])
                continue
            if tag[j][1][:2] == 'RB':
                num_of_adv += 1
                adv_list.append(tag[j][0])
                continue
            if tag[j][1] == 'NN' or tag[j][1] == 'NNS':
                num_of_noun += 1
                continue
            if tag[j][1][:3] == 'PRP':
                num_of_pron += 1
            if tag[j][1] == 'CC':
                num_of_tran += 1
            if j < len(tag) - 2 and tag[j][1] == 'IN' and tag[j+1][1] == 'NN' and tag[j+2][1] == ',':
                num_of_tran += 1

        num_of_distinct_adj = 1 + len(np.unique(np.array(adj_list)))
        num_of_distinct_adv = 1 + len(np.unique(np.array(adv_list)))

        return [
            num_of_verb,
            num_of_adj,
            num_of_adv,
            num_of_pron,
            num_of_tran,
            num_of_noun,
            num_of_distinct_adj,
            num_of_distinct_adv
        ]
