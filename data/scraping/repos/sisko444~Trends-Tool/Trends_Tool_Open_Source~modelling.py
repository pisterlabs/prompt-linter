import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt


class multigrams():
    def __init__(self) :
        import gensim
        return



class nerTagger():

    def __init__(self):
        from nltk.tag import StanfordNERTagger
        from nltk.tokenize import word_tokenize
        import controller as c

        self.st = StanfordNERTagger('C:\\Users\\sisko\\Google Drive\\Sisko Work\\TrendsTool\
        \\stanford-ner-2018-10-16\\classifiers\\english.all.3class.distsim.crf.ser.gz', \
        'C:\\Users\\sisko\\Google Drive\\Sisko Work\\TrendsTool\\stanford-ner-2018-10-16\
        \\stanford-ner.jar', encoding='utf-8')

        import os
        java_path = "C:\\Program Files\\Java\\jre1.8.0_211\\bin\\java.exe"
        os.environ['JAVAHOME'] = java_path

    def tag(self, text) :
        tokenized_text = word_tokenize(text)
        classified_text = self.st.tag(tokenized_text)
        return classified_text

class dataPrep():
    def __init__(self):
        pass
    def preProcessAllData(self, db):

        #import data
        data_words = []
        import re
        for index, alert in db['data'].items() :
            res = ''
            if 'isDuplicate' in alert.keys() : continue
            res += alert['title'] + '\n' +\
            alert['source'] + '\n' +\
            alert['exampleText']
            if not alert['isBroken'] :
                res += '\n' + alert['cleanedHtml']
            res = res.replace('\\', 'x')
            res = re.sub('(xx(\d|\w)(\d|\w)){2,}', '', res)
            # Remove distracting single quotes
            res = re.sub("\'", "", res)
            # Remove newLines
            res = re.sub("\n", "", res)
            res = re.sub("\r", "", res)
            res = sent_to_words(res)
            db['data'][index]['preparedData'] = res
            data_words.append(res)

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        for index, alert in db['data'].items() :
            if 'isDuplicate' in alert.keys() : continue
            bigrams = bigram_mod[alert['preparedData']]
            db['data'][index]['bigrams'] = [x for x in bigrams if '_' in x]
    def topicModel(self, db, indexes, ammountOfTopics) :
        #%matplotlib inline
        # Enable logging for gensim - optional
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
        res = {}
        res['source'] = ''
        res['topics'] = {}

        import warnings
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        # NLTK Stop words
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        stop_words = stop_words + ["error", "link", "http", "unavailable", "http"\
        , "winerror", "host", "attempt", "connection", "site", "remote", "range", \
        "timeout", "request", "googletag", "website", "policy", "security"] + \
        ["able", 'additive', "printing", "manufacturing"\
        , "technology", "also", "print", "cookie"]

        def preProces(db, index):
            alert = db['data'][index]
            res = ''
            res += alert['title'] + '\n' +\
            alert['source'] + '\n' +\
            alert['exampleText']
            if not alert['isBroken'] :
                res += '\n' + alert['cleanedHtml']
            res = res.replace('\\', 'x')
            res = re.sub('(xx(\d|\w)(\d|\w)){2,}', '', res)
            # Remove distracting single quotes
            res = re.sub("\'", "", res)
            # Remove newLines
            res = re.sub("\n", " ", res)
            res = re.sub("\r", "", res)
            return res

        data = []
        for index in indexes:
            if 'isDuplicate' not in db['data'][index].keys() :
                 data.append(preProces(db, index))
        res['source'] += str(len(data)) + ' pieces of data qualified for topic modelling.\n'
        def sent_to_words(sentences):
            for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

        data_words = list(sent_to_words(data))


        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        def remove_stopwords(texts):
            res = []
            for doc in texts :
                res.append([word for word in simple_preprocess(str(doc)) if word not in stop_words])
            return res

        def make_bigrams(texts):
            res = []
            for doc in texts :
                res.append(bigram_mod[doc])
            return res

        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            texts_out = []
            for sent in texts:
                doc = nlp(" ".join(sent))
                texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            return texts_out
            # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = []
        for text in texts :
            corpus.append(id2word.doc2bow(text))

        # Build LDA model
        import os
        os.environ['MALLET_HOME'] = 'C:\\mallet-2.0.8'
        mallet_path = 'C:\\mallet-2.0.8\\bin\\mallet.bat'

        if ammountOfTopics == -1 :
            root = int(round(len(indexes)**(1/3)*1.8))
            res['source'] += 'No given ammount of topics to model so assumed ammount of: ' + str(root) + ' was used.\n'
        else :
            root = ammountOfTopics
        modelTopics = [int(round(root * 0.8)), root, int(round(root * 1.2))]
        coherence = 0
        model = None
        usedTopics = 0
        usedNumber = 0
        topicsToCoherence = []
        modelTopics = list(set(modelTopics))
        for number in modelTopics :
            lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=number, id2word=id2word)
            doc_lda = lda_model[corpus]
            coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            topicsToCoherence.append((number, coherence_lda))
            if coherence_lda > coherence:
                usedTopics = number
                coherence = coherence_lda
                model = lda_model
                usedNumber = number
        if usedNumber == root :
            res['source'] += 'No better ammount of topics to model was found.\n'
        elif usedNumber > root :
            res['source'] += 'A higher ammount of topics was used as it yielded a better coherence score:' + str(usedNumber) + '\n'
        elif usedNumber < root :
            res['source'] += 'A lower ammount of topics was used as it yielded a better coherence score: ' + str(usedNumber) + '\n'
        res['source'] += 'Each number of topic to their coherence: ' + str(topicsToCoherence)
        lda_model = model
        # Get main topic in each document
        def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
            # Init output
            sent_topics_df = pd.DataFrame()

            # Get main topic in each document
            for i, row in enumerate(ldamodel[corpus]):
                row = sorted(row, key=lambda x: (x[1]), reverse=True)
                # Get the Dominant topic, Perc Contribution and Keywords for each document
                for j, (topic_num, prop_topic) in enumerate(row):
                    if j == 0:  # => dominant topic
                        wp = ldamodel.show_topic(topic_num)
                        topic_keywords = ", ".join([word for word, prop in wp])
                        sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                    else:
                        break
            sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

            # Add original text to the end of the output
            contents = pd.Series(texts)
            sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
            return(sent_topics_df)


        df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

        # Group top 5 sentences under each topic
        sent_topics_sorteddf_mallet = pd.DataFrame()

        sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                     grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                                    axis=0)

        # Reset Index
        sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

        # Format
        sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

        # Number of Documents for Each Topic
        topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts/topic_counts.sum(), 4)

        # Topic Number and Keywords
        topic_num_keywords = sent_topics_sorteddf_mallet[['Topic_Num', 'Keywords']]

        # Concatenate Column wise
        df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

        # Change Column names
        df_dominant_topics.columns = ['Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
        #most representative document per topic
        #print(['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"])
        for line in sent_topics_sorteddf_mallet.values.tolist():
            if line[0] not in res['topics'].keys():
                res['topics'][str(line[0])] = 'Topic number: ' + str(line[0]) + ', Match to topic %: ' +\
                 str(round(line[1]*100, 2)) + '\n' + 'Keywords: ' + str(line[2]) + \
                 '\n' + 'Text: \n' + line[3] + '\n'
            else :
                continue
        #document distribution per topic
        #print(['Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents'])
        res['source'] += '\nDisrtibution of data pieces among topics. \n\n '
        for line in df_dominant_topics.values.tolist():
            res['source'] += 'Topic number: ' + str(line[0]) + ', Absolute ammount: ' + str(line[2]) + ', % of total data pieces: ' + str(round(line[3]*100, 2)) + '\n'
            res['source'] += 'Keywords: ' + str(line[1]) + '\n\n'
        return res
if __name__ == '__main__' :
    opr = dataPrep()
    import data
    ins = data.Data(None)
    db = ins.db
    opr.makeCorpus(db)
    ins.save()
