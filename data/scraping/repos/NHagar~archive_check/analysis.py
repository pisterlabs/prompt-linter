from dataclasses import dataclass
import random

from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import spacy
from tqdm import tqdm


random.seed(20210423)

def init_spacy(*, stopwords: "list[str]"=[], disabled: "list[str]"=[]) -> None:
    """Init spacy instance with desired parameters
    """
    nlp = spacy.load("en_core_web_lg", disable=disabled)
    for word in stopwords:
        nlp.vocab[word].is_stop = True
        nlp.vocab[word.title()].is_stop = True

    return nlp

@dataclass
class Table:
    name: str
    df: pd.DataFrame

    def _text_preprocessing(self, doc: spacy.tokens.Doc) -> "list[str]":
        """spacy processing to remove stopwords
        """
        no_punct = [i for i in doc if i.is_alpha]
        no_stop = [i for i in no_punct if not i.is_stop]
        lemmas = [str(i.lemma_).lower() for i in no_stop]
        return lemmas

    def pres_headlines(self):
        """Grab Trump/Biden headline counts
        """
        trump_urls = set(self.df[(self.df.hed.str.contains("Trump"))].url_canonical)
        biden_urls = set(self.df[self.df.hed.str.contains("Biden")].url_canonical)
        both = trump_urls.intersection(biden_urls)
        trump_urls = trump_urls - both
        biden_urls = biden_urls - both
        trump_count = len(trump_urls)
        biden_count = len(biden_urls)

        return trump_count, biden_count

    def count_urls(self) -> int:
        """Count unique URLs in dataframe
        """
        urls = len(self.df.url_canonical.unique())

        return urls

    def process_body(self, nlp: spacy.lang) -> None:
        """Pipeline to process body text with spacy
        """
        preproc_pipe = []
        for doc in tqdm(nlp.pipe(self.df.text, batch_size=20)):
            preproc_pipe.append(self._text_preprocessing(doc))
        self.df.loc[:, "body_parsed"] = preproc_pipe
    
    def build_corpus(self) -> list:
        """Construct corpus for LDA
        """
        tokens = self.df.body_parsed.tolist()
        dictionary_LDA = corpora.Dictionary(tokens)
        lower = round(len(tokens) * 0.005)
        if lower<=1:
            lower = 2
        dictionary_LDA.filter_extremes(no_below=lower, no_above=0.99)
        corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in tokens]

        self.dictionary_lda = dictionary_LDA
        self.corpus = corpus

    def train_models(self, max_k: int) -> "list[dict]":
        """Train LDA models over a range of values
        """
        metrics = []
        for k in tqdm(range(3, max_k+1)):
            lda_model = models.LdaModel(self.corpus, 
                                           num_topics=k,
                                           id2word=self.dictionary_lda,
                                           passes=5, 
                                           alpha="auto",
                                           eta="auto")
            cm = CoherenceModel(model=lda_model,
                                corpus=self.corpus,
                                dictionary=self.dictionary_lda,
                                coherence="u_mass")
            coherence = cm.get_coherence()
            topics = lda_model.show_topics(num_topics=k)
            topic_string = "\n".join([f"Topic: {i[0]}, Tokens: {i[1]}" for i in topics])
            result = {
                "k": k,
                "words": topic_string,
                "coherence": coherence
            }
            metrics.append(result)
        
        self.metrics = metrics

    def get_best_model(self) -> dict:
        """Select optimal model from trained range
        """
        best = sorted(self.metrics, key=lambda x: x['coherence'])[-1]
        best['service'] = self.name

        return best