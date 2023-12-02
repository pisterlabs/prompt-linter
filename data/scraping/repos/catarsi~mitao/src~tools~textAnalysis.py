from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import string
from operator import itemgetter

# Importing Gensim
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

import pandas as pd


class TextAnalysis(object):

    def __init__(self):
        pass

    def tokenize(self, input_files, param, tool_id):

        data_to_return = {"data":{}}
        ok_to_process = False

        # Check the tool needs
        # -----
        if "d-gen-text" in input_files:
            if len(input_files["d-gen-text"]):
                ok_to_process = True

        if not ok_to_process:
            res_err = {"data":{}}
            res_err["data"]["error"] = "Input data missing!"
            return res_err
        # -----

        # Params
        # -----
        p_stopwords = "none" #string e.g. "English"
        p_lemmatize_lang = "none"
        if param != None:
            if "p-defstopwords" in param:
                p_stopwords = str(param["p-defstopwords"])
            if "p-deflemmatize" in param:
                p_lemmatize_lang = str(param["p-deflemmatize"])


        # Data (Input Documents)
        # use pandas and convert to DataFrame
        # -----
        documents = {}
        for file_k in input_files["d-gen-text"]:
            documents[file_k] =  input_files["d-gen-text"][file_k]
        docs_df = pd.DataFrame.from_dict(documents, orient='index', columns=["content"])


        def read_csv_rows(obj_data):
            res = []
            for file_name in obj_data:
                for a_row in obj_data[file_name]:
                    for a_val in a_row:
                        #normalize and append
                        a_val = a_val.lower()
                        a_val = a_val.strip()
                        res.append(a_val)
            return res

        stopwords_data = set()
        if "d-stopwords" in input_files:
            if len(input_files["d-stopwords"]) > 0:
                for f in input_files["d-stopwords"]:
                    stopwords_data = stopwords_data.union(set(read_csv_rows(input_files["d-stopwords"])))

        tokens_data = set()
        if "d-tokens" in input_files:
            if len(input_files["d-tokens"]) > 0:
                for f in input_files["d-tokens"]:
                    tokens_data = tokens_data.union(set(read_csv_rows(input_files["d-tokens"])))

        if p_stopwords != "none":
            stopwords_data = stopwords_data.union(set(stopwords.words(p_stopwords)))

        def lemmatize_stemming(text,lang):
            #lemmatize only if the token includes one word
            if (" " not in text) and ("-" not in text):
                stemmer = SnowballStemmer(lang)
                return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
            else:
                return text

        def adhoc_tokens_list(l_token):
            if len(l_token) == 1:
                token = l_token[0]
                if "-" in token:
                    new_l = []
                    for elem in token.split("-"):
                        if elem != "":
                            new_l.append(elem)
                    return adhoc_tokens_list(new_l)
                elif " " in token:
                    new_l = []
                    for elem in token.split(" "):
                        if elem != "":
                            new_l.append(elem)
                    return adhoc_tokens_list(new_l)
                else:
                    return l_token
            else:
                mid = len(l_token)//2
                return adhoc_tokens_list(l_token[:mid]) + adhoc_tokens_list(l_token[mid:])


        def update_tokenlist(org_token_l, adhoc_token_l, adhoc_token):
            index = 0
            res = []
            while (len(org_token_l) - index) >= len(adhoc_token_l):
                part_of_org = org_token_l[index:index+len(adhoc_token_l)]
                if adhoc_token_l == part_of_org:
                    #res += part_of_org + [adhoc_token] #in case i want to include the single words
                    res.append(adhoc_token) #in case i want to include only the adhoc token
                    index += len(adhoc_token_l)
                else:
                    res.append(org_token_l[index])
                    index += 1
            res += org_token_l[index:]
            return res

        # giving a text this function
        # (1) creates a list with all the single words ;
        #(2) includes the ad-hoc defined tokens;
        #(3) lemmatizes and removes the stopwords
        def tokenize_text(text):
            result = []
            #add the automatic detected tokens
            for token in gensim.utils.simple_preprocess(text):
                result.append(token)

            #add the ad-hoc tokens
            for ad_hoc_token in tokens_data:
                adhoc_token_l = adhoc_tokens_list([ad_hoc_token])
                result = update_tokenlist(result, adhoc_token_l, ad_hoc_token)

            #remove stopwords and create lemmatize form
            clean_result = []
            for token in result:
                if token not in stopwords_data and len(token) > 3:
                    if p_lemmatize_lang != "none":
                        clean_result.append(lemmatize_stemming(token,p_lemmatize_lang))
                    else:
                        clean_result.append(token)

            return clean_result

        processed_docs = docs_df['content'].map(tokenize_text)
        processed_docs_ldict = []
        for k,doc in processed_docs.items():
            processed_docs_ldict.append({"index":k,"value":doc})

        data_to_return["data"]["d-processed-corpus"] = {"processed_corpus": processed_docs_ldict}
        return data_to_return

    def build_corpus(self, input_files, param, tool_id):

        data_to_return = {"data":{}}
        ok_to_process = False
        #Check the MUST Prerequisite
        # Check Restrictions
        if "d-processed-corpus" in input_files:
            if len(input_files["d-processed-corpus"]):
                ok_to_process = True

        if not ok_to_process:
            res_err = {"data":{}}
            res_err["data"]["error"] = "Input data missing!"
            return res_err

        # convert to data series
        indexes = []
        values = []
        # in this case i expect only 1 file
        for file_k in input_files["d-processed-corpus"]:
            for d in input_files["d-processed-corpus"][file_k]:
                indexes.append(d["index"])
                values.append(d["value"])
        processed_docs = pd.Series(values, index =indexes)

        #The params
        #---------
        p_model = None #string e.g. "English"
        if param != None:
            if "p-corpusmodel" in param:
                p_model = str(param["p-corpusmodel"])

        # -> Create the dictionary of words containing the number of times a word appears in the training set
        # -> Filter out tokens that appear in: (a) less than 15 documents; OR (b) more than 0.5 documents
        # -> and keep only the first 100000 most frequent tokens.
        # -----
        dictionary = gensim.corpora.Dictionary(processed_docs)
        #dictionary.filter_extremes()
        index_corpus = []
        vec_corpus = []
        for k,doc in processed_docs.items():
            vec_corpus.append(dictionary.doc2bow(doc))
            index_corpus.append(k)

        # TF-IDF
        if p_model == "tfidf":
            tfidf = models.TfidfModel(vec_corpus)
            vec_corpus = tfidf[vec_corpus]


        #The returned data must include a recognizable key and the data associated to it
        # -----
        vec_corpus_ldict = []
        for i in range(0,len(vec_corpus)):
            vec_corpus_ldict.append({"index":index_corpus[i],"value":vec_corpus[i]})

        data_to_return["data"]["d-model-corpus"] = {"modelled_corpus": vec_corpus_ldict}
        data_to_return["data"]["d-dictionary-corpus"] = {"dictionary": dictionary}
        return data_to_return

    def lda(self, input_files, param, tool_id):

        data_to_return = {"data":{}}
        ok_to_process = False

        # Check the tool needs
        # -----
        if "d-model-corpus" in input_files and "d-dictionary-corpus" in input_files:
            ok_to_process = len(input_files["d-model-corpus"]) and len(input_files["d-dictionary-corpus"])

        if not ok_to_process:
            res_err = {"data":{}}
            res_err["data"]["error"] = "Input data missing!"
            return res_err

        corpus = []
        for file_k in input_files["d-model-corpus"]:
            for d in input_files["d-model-corpus"][file_k]:
                corpus.append(d["value"])

        dictionary = None
        for file_k in input_files["d-dictionary-corpus"]:
            dictionary = input_files["d-dictionary-corpus"][file_k]

        # Params
        # -----
        p_num_topics = 2 #int number
        if param != None:
            if "p-topic" in param:
                p_num_topics = int(param["p-topic"])

        # Running LDA
        # -----
        try:
            ldamodel = gensim.models.LdaMulticore(corpus, eval_every = 1, num_topics=p_num_topics, id2word=dictionary, passes=5, workers=2)
        except:
            res_err = {"data":{}}
            res_err["data"]["error"] = "Incompatible data have been given as input to the LDA algorithm"
            return res_err

        data_to_return["data"]["d-gensimldamodel"] = {"ldamodel": ldamodel}
        return data_to_return

    def doc_prop_topics(self, input_files, param, tool_id):

        data_to_return = {"data":{}}
        ok_to_process = False

        # Check the tool needs
        # -----
        if "d-model-corpus" in input_files and "d-gensimldamodel" in input_files:
            ok_to_process = len(input_files["d-model-corpus"]) and len(input_files["d-gensimldamodel"])

        if not ok_to_process:
            res_err = {"data":{}}
            res_err["data"]["error"] = "Input data missing!"
            return res_err

        corpus = []
        corpus_doc_index = []
        for file_k in input_files["d-model-corpus"]:
            for d in input_files["d-model-corpus"][file_k]:
                corpus.append(d["value"])
                corpus_doc_index.append(d["index"])

        ldamodel = None
        for file_k in input_files["d-gensimldamodel"]:
            ldamodel = input_files["d-gensimldamodel"][file_k]

        # Params
        # -----

        def _doc_topics(ldamodel, corpus, corpus_doc_index):

            ## doc_topics_l -> [ [0.23,0.4, ... <num_topics> ] [] []  ... []]
            doc_topics = ldamodel.get_document_topics(corpus, minimum_probability=0)
            doc_topics_l = []
            for l in doc_topics:
                doc_topics_l.append([tup[1] for tup in l])

            return pd.DataFrame(doc_topics_l, columns = list(range(1, ldamodel.num_topics + 1)), index = corpus_doc_index)

        df_doc_topics = _doc_topics(ldamodel,corpus, corpus_doc_index)
        df_doc_topics.index.names = ['doc']
        df_doc_topics = df_doc_topics.reset_index()
        l_doc_topics = [df_doc_topics.columns.values.tolist()] + df_doc_topics.values.tolist()

        data_to_return["data"]["d-doc-topics-table"] = {"doc_topics": l_doc_topics}
        return data_to_return

    def words_prop_topics(self, input_files, param, tool_id):

        data_to_return = {"data":{}}
        ok_to_process = False

        # Check the tool needs
        # -----
        if "d-model-corpus" in input_files and "d-gensimldamodel" in input_files:
            ok_to_process = len(input_files["d-model-corpus"]) and len(input_files["d-gensimldamodel"])

        if not ok_to_process:
            res_err = {"data":{}}
            res_err["data"]["error"] = "Input data missing!"
            return res_err

        corpus = []
        corpus_doc_index = []
        for file_k in input_files["d-model-corpus"]:
            for d in input_files["d-model-corpus"][file_k]:
                corpus.append(d["value"])
                corpus_doc_index.append(d["index"])

        ldamodel = None
        for file_k in input_files["d-gensimldamodel"]:
            ldamodel = input_files["d-gensimldamodel"][file_k]

        # Params
        # -----
        topnum_words = 10 #int number
        if param != None:
            if "p-numwords" in param:
                topnum_words = int(param["p-numwords"])

        def _word_topics(ldamodel, corpus, corpus_doc_index):

            topics = []
            for t_index in range(0, ldamodel.num_topics):
                wp = ldamodel.show_topic(t_index, topn=topnum_words)
                topic_keywords = [[t_index + 1,word,prop] for word, prop in wp]
                topic_keywords = sorted(topic_keywords, key=itemgetter(2), reverse=True)
                topics = topics + topic_keywords

            return pd.DataFrame(topics, columns = ["topic","word","prop"])

        df_topics = _word_topics(ldamodel,corpus, corpus_doc_index)
        l_topics = [df_topics.columns.values.tolist()] + df_topics.values.tolist()
        data_to_return["data"]["d-word-topics-table"] = {"word_topics": l_topics}
        return data_to_return

    def calc_coherence(self, input_files, param, tool_id):

        data_to_return = {"data":{}}

        # Pre
        # -------
        # Check Restrictions
        ok_to_process = "d-model-corpus" in input_files and "d-dictionary-corpus" in input_files
        ok_to_process = ok_to_process and (len(input_files["d-model-corpus"]) == 1  and len(input_files["d-dictionary-corpus"]) == 1)
        if not ok_to_process:
            data_to_return["data"]["error"] = "unexpected or missing input!"
            return data_to_return

        # Read inputs
        inputs = {"victorized_corpus": [], "g_dict": {}}
        for file_k in input_files["d-model-corpus"]:
            for d in input_files["d-model-corpus"][file_k]:
                inputs["victorized_corpus"].append(d["value"])

        for file_k in input_files["d-dictionary-corpus"]:
            inputs["g_dict"] =  input_files["d-dictionary-corpus"][file_k]

        # Params
        NUM_TOPICS_FROM = None
        NUM_TOPICS_TO = None
        if param != None:
            if "p-topic-from" in param:
                NUM_TOPICS_FROM = int(param["p-topic-from"])
            if "p-topic-to" in param:
                NUM_TOPICS_TO = int(param["p-topic-to"])


        # Process
        # -------
        coherenceList_umass = [["topics","score"]]
        for num_topics in range(NUM_TOPICS_FROM,NUM_TOPICS_TO+1):
            ldamodel = gensim.models.LdaMulticore(inputs["victorized_corpus"], num_topics=num_topics, id2word=inputs["g_dict"], passes=5, workers=2)
            cm = CoherenceModel(model=ldamodel, corpus=inputs["victorized_corpus"], dictionary=inputs["g_dict"], coherence='u_mass')
            coherenceList_umass.append([num_topics,cm.get_coherence()])

        data_to_return["data"]["d-coherence"] = {'coherence': coherenceList_umass}
        return data_to_return
