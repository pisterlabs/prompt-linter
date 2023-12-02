import plotly.express as px
from sklearn.base import copy
from gensim.test.utils import common_dictionary
from gensim.models import CoherenceModel
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display

class TopicVisualizer:
    def __init__(self, topic_pipe, df_texts, text_col, date_col):
        """
        Parameters:
        ----------
        topic_pipe: sklearn.pipeline.Pipeline
            Fitted topic pipeline containing steps: `preprocessor`, `vectorizer`, `model`.
        df_text: pd.DataFrame
            
        """
        self.pipe = topic_pipe
        self.df_texts = df_texts
        self.text_col = text_col
        self.date_col = date_col
        
        self.transform()
        
    def transform(self):
        """Transforms nested `df_texts` storing all intermediate steps."""
        self.texts_prep = self.pipe.named_steps.preprocessor.transform(self.df_texts[self.text_col])
        self.feat = self.pipe.named_steps.vectorizer.transform(self.texts_prep)
        self.data_topics = self.pipe.named_steps.model.transform(self.feat)
        return self.df_topics
    
    
    @staticmethod
    def _plot_top_words(model, feature_names, n_top_words, title):
        n_components = len(model.components_)
        fig, axes = plt.subplots(n_components // 5, 5, figsize=(30, 1.5 * n_components), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
            ax.invert_yaxis()
            ax.tick_params(axis="both", which="major", labelsize=20)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show()
        
    @staticmethod
    def _get_top_words(model, feature_names, n_top_words, join=' | '):
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            if join is None:
                yield top_features
            else:
                yield join.join(top_features)
                
    def calculate_coherence_from_n_topics(self, n_components_it=[2, 5, 10, 15, 20, 25, 30], coherence='c_v'):
        dictionary = common_dictionary.from_documents(self.texts_prep)
        
        display(dictionary)

        scores = {}
        for n_components in tqdm(n_components_it):
            model = copy.copy(self.pipe.named_steps.model)
            model.n_components = n_components
            model.fit(self.feat)
            topics = self._get_top_words(model, self.pipe.named_steps.vectorizer.get_feature_names(), 5, None)

            cm = CoherenceModel(
                topics=topics,
                dictionary=dictionary,
                texts=self.texts_prep,
                coherence=coherence
            )
            scores[n_components] = cm.get_coherence()
        return scores
    
    def plot_coherence_from_n_topics(self, n_components_it=[2, 5, 10, 15, 20, 25, 30], coherence='c_v'):
        scores_dct = self.calculate_coherence_from_n_topics(n_components_it, coherence)
        scores_ser = pd.Series(scores_dct, name='coherence')
        return px.line(scores_ser, title=f'Coherence "{coherence}" by number of topics')
    
    def plot_top_keywords(self, n_words=20, title='Top words'):
        return self._plot_top_words(
            self.pipe.named_steps.model,
            self.pipe.named_steps.vectorizer.get_feature_names(),
            n_words,
            title)
    
    def get_top_words(self, n_words=5, join=' | '):
        return list(self._get_top_words(
            self.pipe.named_steps.model,
            self.pipe.named_steps.vectorizer.get_feature_names(),
            n_words,
            join
        ))
    
    @property
    def df_topics(self):
        return pd.DataFrame(
            self.data_topics,
            columns=self.get_top_words(n_words=3),
            index=self.df_texts.index
        )
    
    @property
    def df_top_topic_for_doc(self):
        return self.df_topics.agg(['idxmax', 'max'], axis=1).sort_values('max').join(self.df_texts)
    
    @property
    def df_top_doc_for_topic(self):
        return self.df_topics.agg(['max', 'idxmax']).T.merge(self.df_texts, left_on='idxmax', right_index=True).rename(columns={'max': 'weight'})

    def plot_topic_trend(self, min_score=0.2):
        df_topics_by_date_gr = (self.df_topics[self.df_topics > min_score]
                                .join(self.df_texts[self.date_col])
                                .rename_axis(columns='topic')
                                .groupby(
                                    pd.Grouper(key=self.date_col, freq='m')
                                )
                               )

        return px.line(
            df_topics_by_date_gr.count().stack().to_frame('count').reset_index(),
            x=self.date_col,
            y='count',
            facet_col='topic',
            facet_col_wrap=3,
            height=900,
        )
    
    def plot_doc_by_top_topic(self, text_col):
        text_col = text_col or self.text_col
        return px.box(
            self.df_top_topic_for_doc,
            facet_col='idxmax',
            facet_col_wrap=3,
            x='max',
            points='all',
            hover_data=[text_col],
            height=800
        )

    def plot_topic_weight_distribution(self, **kwargs):
        default_kwargs = dict(x='weight', log_y=True, facet_col='topic', height=900, facet_col_wrap=3)
        default_kwargs.update(kwargs)
        return px.histogram(self.df_topics.stack().to_frame('weight').reset_index().query('weight > 0').rename(columns={'level_1': 'topic'}), **default_kwargs)