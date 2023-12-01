import re

import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models import HdpModel

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import ngrams
from nltk.corpus import stopwords

### NLP data processing
class ClinicalNoteLDA():
    """
    Processes set of clinical notes into LDA model
    """
    
    def __init__(self, texts:list):
        # preprocessing
        self.texts = texts
        
        # run lda_model()
        self.dictionary=None
        self.corpus = None
        self.lda_model = None
        self.coherence = None
        
        self.vis = None

    def create_lda(self, no_below=5, no_above=0.5, num_topics=10, chunksize=2000, 
                     alpha='auto', per_word_topics=True, coherence='u_mass',**kwargs):
        """
        LDA model of digital health usage
        """
        # Term Document Frequency
        self.dictionary = Dictionary(self.texts)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, **kwargs)

        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        
        # LDA model
        self.lda_model = LdaModel(corpus=self.corpus, id2word=self.dictionary, random_state=0, 
                                  num_topics=num_topics, alpha=alpha, chunksize=chunksize, 
                                  per_word_topics=per_word_topics,  **kwargs) 
        
        cm = CoherenceModel(model=self.lda_model, texts=self.texts, dictionary=self.dictionary, coherence=coherence)
        self.coherence = cm.get_coherence()  # get coherence value
        
    def visualize_lda(self):
        """Visualize LDA"""
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim_models.prepare(topic_model=self.lda_model, corpus=self.corpus, dictionary=self.dictionary, sort_topics=True)
        return vis
    
    def hparam_sweep(self, grid_search, no_below=5, no_above=0.5, coherence="u_mass"):
        """
        Grid search for number of topics
        # TODO: figure out lda_model.log_perplexity(corpus)
        
        Parameters
        ---------
        grid_search (dict): {"num_topics":<list<int>>}
        """
        coherence_values = {}
        for t in grid_search["num_topics"]:
            self.create_lda(no_below=no_below, no_above=no_above, num_topics=t, coherence=coherence)
            coherence_values[f"{t}"] = self.coherence

        coherence_df = pd.DataFrame.from_dict(coherence_values, orient="index").reset_index()
        fig, ax = plt.subplots(figsize=(10,10))
        sns.scatterplot(data=coherence_df,  y=0, x="index") 
        return coherence_df
    
    def preprocessDHNotes(self, custom_stopwords=[]):
        """Preprocess notes and tokenize"""
        tokenizer = RegexpTokenizer(r'\w+')
        all_stopwords = list(stopwords.words('english'))
        all_stopwords = all_stopwords + custom_stopwords

        dh_texts = [t.replace("\n", ". ") for t in self.texts]
        dh_texts = [re.sub('[^\sA-Za-z]+', ' ', i) for i in dh_texts] 

        dh_texts = [tokenizer.tokenize(t) for t in dh_texts]
        dh_texts = [[t for t in text if len(t)>1 if t not in all_stopwords] for text in dh_texts]
        self.texts = dh_texts
    
    @staticmethod
    def scatterplot_topics(vis, figsize=(11,11), scale=20000,
                           x_axis_lim=[-0.32, 0.35], y_axis_lim=[-0.42, 0.3], ):
        """
        Recreate gensim scatterplot just with easier to customize values
        
        Parameters
        ---------
            vis (pyLDAvis object): The visualization object containing topic coordinates.
            figsize (int,int): The size of the figure (width, height) in inches. Default is (11, 11).
            scale (int): The scaling factor for the size of the scatterplot markers. Default is 20000.
            x_axis_lim (float,float): The limits for the x-axis range. Default is [-0.32, 0.35].
            y_axis_lim (float,float): The limits for the y-axis range. Default is [-0.42, 0.3].
        Returns
        ---------
            ax: The matplotlib Axes object containing the scatterplot.
        
        """
        fig, ax = plt.subplots(figsize=figsize)
        topic_values = vis.topic_coordinates
        topic_values["scale"] = topic_values["Freq"] / topic_values["Freq"].abs().max()
        topic_values["scale"] = topic_values["scale"] * scale
        ax = sns.scatterplot(data=topic_values, 
                             x="x", y="y", 
                             size="Freq", 
                             alpha=0.3, 
                             sizes=(min(topic_values["scale"]), max(topic_values["scale"])), 
                             linewidth=3,
                             edgecolor="darkgrey",
                             legend=False)

        for line in range(0,topic_values.shape[0]):
             ax.text(topic_values.x[line], topic_values.y[line], 
                             topic_values.topics[line], horizontalalignment='center', va="center",
                     size='small', color='black', weight='normal')

        ax.set_xlim(x_axis_lim)
        ax.set_ylim(y_axis_lim)

        ax.set_xlabel("PC0")
        ax.set_ylabel("PC1")

        ax.set_title("LDA Intertopic Distance Map")

        return ax
    
    @staticmethod
    def barplot_top_terms(vis, top_n=10, normalize=True, text_size='small'):
        """
        Generates barplots of top words in each topic
        Parameters
        ---------
        vis (pyLDAvis object): pyLDAvis object
        top_n (int): The number of top words to display in each topic. Default is 10.
        normalize (bool): Plots estimated percentage of term occurence in each topic, plots raw counts if False
        text_size (str): The size of the text in the plot. Options: 'small', 'medium', 'large'. Default is 'small'.
        """
        term_freq = vis.topic_info
        term_freq = term_freq.sort_values("Freq", ascending=False).groupby("Category").head(top_n)
        term_freq = term_freq[term_freq["Category"]!="Default"]
        term_freq = term_freq.sort_values(["Category", "Freq"], ascending=[True, False])
        term_freq["Frequency"] = term_freq["Freq"]

        if normalize:
            term_freq["Sum"] = term_freq.groupby("Category")["Freq"].transform('sum')
            term_freq["Frequency"] = term_freq["Frequency"] / term_freq["Sum"]

        g = sns.catplot(data=term_freq, x="Frequency", y="Term", col="Category",
                         kind="bar", height=6, aspect=0.7, col_wrap=5, 
                         sharey=False, sharex=False, facet_kws=dict(margin_titles=True), )

        g.figure.subplots_adjust(wspace=.55, hspace=.2)
        g.set_titles(template="{col_name}")

        return g

    @staticmethod
    def get_top_topic(topic_probs, min_prob=None):
        """
        Return the top topic given a document score calculated by lda_model.get_document_topics
        topic_probs (list<(int,float)>): list of topics and probabilities
        min_prob (float): only take values that have predicted probability above this value, else return -1
        """
        max_val = topic_probs[0]
        for prob in topic_probs:
            if prob[1]>max_val[1]:
                max_val = prob
        
        if min_prob is not None:
            if max_val[1] < min_prob: 
                max_val=(-1,max_val[0])
            
        return max_val