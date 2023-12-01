# Copyright (c) 2023 Aptima, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
import pickle
import re
from pprint import pprint

import gensim
import gensim.corpora as corpora
import pandas as pd
import spacy
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from tqdm import tqdm

from yaada.core import default_log_level, schema
from yaada.core.analytic import YAADAAnalytic
from yaada.core.analytic.model import ModelBase
from yaada.nlp.utils import ensure_nltk_stopwords, ensure_spacy_model

logger = logging.getLogger(__name__)
logger.setLevel(default_log_level)


def doc2words(doc):
    return simple_preprocess(str(doc), deacc=True)  # deacc=True removes punctuations


def clean_text(text):
    # cleaning the data
    try:
        text = re.sub(r"\S*@\S*\s?", "", text)

        # Remove new line characters
        text = re.sub(r"\s+", " ", text)

        # Remove distracting single quotes
        text = re.sub(r"\'", "", text)
    except TypeError:
        logger.error(f"error cleaning '${text}'", exc_info=True)
        return None

    return text


# # Define functions for stopwords, bigrams, trigrams and lemmatization
# def remove_stopwords(doc_words,stop_words):
#     return [word for word in doc_words if word not in stop_words]


class GensimLDAModel(ModelBase):
    def __init__(self, model_instance_id):
        super(GensimLDAModel, self).__init__(model_instance_id)
        ensure_spacy_model("en_core_web_sm")
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def remove_stopwords(self, doc_words):
        return [word for word in doc_words if word not in self.stop_words]

    def make_bigrams(self, doc_words):
        return self.bigram_mod[doc_words]

    def make_trigrams(self, doc_words):
        return self.trigram_mod[self.bigram_mod[doc_words]]

    def lemmatize(self, doc_words, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
        """https://spacy.io/api/annotation"""

        doc = self.nlp(" ".join(doc_words))
        return [
            token.lemma_
            for token in doc
            if token.pos_ in allowed_postags and token.lemma_ not in self.stop_words
        ]

    def build(self, df, parameters):
        ensure_nltk_stopwords()
        from nltk.corpus import stopwords

        # analyze_doc_type = parameters["analyze_doc_type"]
        analyze_feature = parameters["analyze_feature"]
        stop_words = stopwords.words("english") + parameters.get("stop_words", [])
        num_topics = parameters.get("num_topics", 10)
        random_state = parameters.get("random_state", 100)
        update_every = parameters.get("update_every", 1)
        chunksize = parameters.get("chunksize", 100)
        passes = parameters.get("passes", 10)

        # Initialize spacy 'en_core_web_sm' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en_core_web_sm

        self.stop_words = stop_words

        # cleaning the data
        data = df[analyze_feature].values.tolist()

        data = [doc for doc in data if doc]

        data = [clean_text(doc) for doc in tqdm(data, desc="cleaning data")]

        data_words = [doc2words(text) for text in tqdm(data, desc="doc2words") if text]

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(
            tqdm(data_words, desc="bigrams"), min_count=5, threshold=100
        )  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(
            tqdm(bigram[data_words], desc="trigrams"), threshold=100
        )

        # Faster way to get a sentence clubbed as a trigram/bigram
        self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(trigram)

        # Remove Stop Words
        data_words_nostops = [
            self.remove_stopwords(doc)
            for doc in tqdm(data_words, desc="remove stopwords")
        ]

        # Form Bigrams
        data_words_bigrams = [
            self.make_bigrams(doc)
            for doc in tqdm(data_words_nostops, desc="bigrams after applying stopwords")
        ]

        # Do lemmatization keeping only noun, adj, vb, adv
        self.data_lemmatized = [
            self.lemmatize(doc, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"])
            for doc in tqdm(data_words_bigrams, desc="lemmatize")
        ]

        # Create Dictionary
        self.id2word = corpora.Dictionary(
            tqdm(self.data_lemmatized, desc="create dictionary")
        )

        # Create Corpus
        texts = self.data_lemmatized

        # Term Document Frequency
        self.corpus = [
            self.id2word.doc2bow(text) for text in tqdm(texts, desc="doc2bow")
        ]

        print("Training model")
        self.lda_model = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=num_topics,
            random_state=random_state,
            update_every=update_every,
            chunksize=chunksize,
            passes=passes,
            alpha="auto",
            per_word_topics=True,
        )

    def perplexity(self):
        return self.lda_model.log_perplexity(self.corpus)

    def coherence(self):
        coherence_model_lda = CoherenceModel(
            model=self.lda_model,
            texts=self.data_lemmatized,
            dictionary=self.id2word,
            coherence="c_v",
        )
        return coherence_model_lda.get_coherence()

    def save_artifacts(self, path):
        ldamodel_dir = self.make_artifact_dir(path, artifact_type="LDAModel")

        self.id2word.save(ldamodel_dir + "/id2word")
        self.lda_model.save(ldamodel_dir + "/lda_model")
        pickle.dump(
            self.data_lemmatized, open(ldamodel_dir + "/data_lemmatized.p", "wb")
        )
        pickle.dump(self.corpus, open(ldamodel_dir + "/corpus.p", "wb"))
        pickle.dump(self.stop_words, open(ldamodel_dir + "/stop_words.p", "wb"))
        self.bigram_mod.save(ldamodel_dir + "/bigram_mod")
        self.trigram_mod.save(ldamodel_dir + "/trigram_mod")

        # data = dict(content=self.content)
        # with open(os.path.join(resource_path,'content.json'),'w') as f:
        #   json.dump(data,f)

    def load_artifacts(self, path):
        ldamodel_dir = self.get_artifact_dir(path, artifact_type="LDAModel")

        self.id2word = corpora.Dictionary.load(ldamodel_dir + "/id2word")
        self.lda_model = gensim.models.ldamodel.LdaModel.load(
            ldamodel_dir + "/lda_model"
        )
        self.data_lemmatized = pickle.load(
            open(ldamodel_dir + "/data_lemmatized.p", "rb")
        )
        self.corpus = pickle.load(open(ldamodel_dir + "/corpus.p", "rb"))
        self.stop_words = pickle.load(open(ldamodel_dir + "/stop_words.p", "rb"))
        self.bigram_mod = gensim.models.phrases.Phraser.load(
            ldamodel_dir + "/bigram_mod"
        )
        self.trigram_mod = gensim.models.phrases.Phraser.load(
            ldamodel_dir + "/trigram_mod"
        )
        # with open(os.path.join(resource_path,'content.json'),'r') as f:
        #   data = json.load(f)
        #   self.build(data['content'])

    def print_topics(self):
        pprint(self.lda_model.print_topics())

    def topics(self, num_words=20):
        return self.lda_model.show_topics(formatted=False, num_words=num_words)

    def text2bow(self, text):
        cleaned_text = clean_text(text)
        words = doc2words(cleaned_text)
        # Remove Stop Words
        words_nostops = self.remove_stopwords(words)

        bigrams = self.make_bigrams(words_nostops)

        lemmatized = self.lemmatize(
            bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
        )

        return self.id2word.doc2bow(lemmatized)

    def topics_for_text(
        self,
        text,
        minimum_probability=None,
        minimum_phi_value=None,
        per_word_topics=False,
    ):
        bow = self.text2bow(text)

        topics = self.lda_model.get_document_topics(
            bow,
            minimum_probability=minimum_probability,
            minimum_phi_value=minimum_phi_value,
            per_word_topics=per_word_topics,
        )

        topics = [(t[0], float(t[1])) for t in topics]

        return topics


class GensimTrainLDAModel(YAADAAnalytic):
    DESCRIPTION = "Train an LDA topic model."
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "model_instance_id": {
                "description": "id for the trained model",
                "type": "string",
            },
            "analyze_doc_type": {
                "description": "the type of document to analyze",
                "type": "string",
            },
            "analyze_feature": {
                "description": "the name of the text field on the document to use for analysis",
                "type": "string",
            },
            "analyze_query": {
                "description": "the elasticsearch query for fetching documents to analyze",
                "type": "object",
            },
            "analyze_query_size": {
                "description": "the number of results to return",
                "type": "number",
            },
            "stop_words": {
                "description": "additional stopwords",
                "type": "array",
                "items": {"type": "string"},
            },
            "num_topics": {"description": "", "type": "number"},
            "random_state": {"description": "", "type": "number"},
            "update_every": {"description": "", "type": "number"},
            "chunksize": {"description": "", "type": "number"},
            "passes": {"description": "", "type": "number"},
        },
        "required": ["analyze_doc_type", "analyze_feature", "model_instance_id"],
    }
    REQUEST_SCHEMA = schema.make_request_schema(PARAMETERS_SCHEMA)

    def __init__(self):
        pass

    def run(self, context, request):
        model_instance_id = request["parameters"]["model_instance_id"]

        analyze_doc_type = request["parameters"]["analyze_doc_type"]
        analyze_feature = request["parameters"]["analyze_feature"]

        analyze_query_size = request["parameters"].get("analyze_query_size", None)

        q = {
            "query": {"match_all": {}},
            "_source": {"include": ["doc_type", "_id", analyze_feature]},
        }

        if "analyze_query" in request["parameters"]:
            q = request["parameters"]["analyze_query"]

        # load into dataframe
        #
        modelmanager = context.get_model_manager()
        model = modelmanager.load_model_instance(GensimLDAModel, model_instance_id)
        if model is None:
            model = GensimLDAModel(model_instance_id)

        df = pd.DataFrame.from_records(
            list(
                tqdm(
                    context.query(analyze_doc_type, q, size=analyze_query_size),
                    desc="fetching",
                )
            ),
            index="_id",
        )

        model.build(df, request["parameters"])
        modelmanager.save_model_instance(model)
