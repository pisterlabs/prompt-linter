import argparse
import os
from azureml.core import Run
import pandas as pd
import numpy as np

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim_models
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime


class TopicModelling:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.df = None
        self.df_topics = pd.DataFrame()
        self.df_document_topic = None

        self.corpus = None
        self.corpus_doc_words = None
        self.corpus_dict = None
        self.lda_model = None

        self.get_runtime_arguments()

        self.load_dataset()

        self.topic_modelling()

        self.plot_topics_temporal_evolution()

        self.output_datasets()

    @staticmethod
    def text_to_words_cleansed(doc_text):
        for token in doc_text:
            yield gensim.utils.simple_preprocess(str(token), deacc=True)

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--input',
            type=str,
            help='Input extract data'
        )
        parser.add_argument(
            '--n_topics',
            type=int,
            help='Number of topics to model'
        )
        parser.add_argument(
            '--n_top_words',
            type=int,
            help='N top words per topic'
        )
        parser.add_argument(
            '--optimise',
            dest='optimise', action='store_true',
            help='Optimise by coherence score'
        )
        parser.add_argument(
            '--no_optimise',
            dest='optimise', action='store_false',
            help='Do not optimise by coherence score'
        )
        parser.set_defaults(feature=True)

        parser.add_argument(
            '--output_topics',
            type=str,
            help=' Output topics'
        )
        parser.add_argument(
            '--output_document_topic',
            type=str,
            help=' Output document topic'
        )

        self.args = parser.parse_args()

        print('Input: {}'.format(self.args.input))
        print('N topics: {}'.format(self.args.n_topics))
        print('N top words: {}'.format(self.args.n_top_words))
        print('Optimise by coherence score: {}'.format(self.args.optimise))
        print('Output topics: {}'.format(self.args.output_topics))
        print('Output document topic: {}'.format(self.args.output_document_topic))

    def load_dataset(self):
        print('--- Load Data')
        path = self.args.input + "/processed.csv"
        self.df = pd.read_csv(path, dtype={
            'hash_id': str,
            'paper_id': str,
            'processed_title': str,
            'processed_abstract': str})

        print('Raw Input Specifications')
        print(self.df.columns)
        print(self.df.shape)

        print('Input Following Column Subset')
        self.df = self.df[['hash_id', 'paper_id', 'processed_title', 'processed_abstract', 'publish_time']]
        print(self.df.columns)
        print(self.df.shape)

    def set_topic_words(self):
        print('--- Set topic words')
        topic_collection = []

        for idx, topic_words in self.lda_model.show_topics(formatted=False, num_words=self.args.n_top_words):
            for tw in topic_words:
                topic_collection.append([idx, tw[0], round(tw[1], 5)])

        self.df_topics = pd.DataFrame(topic_collection, columns=['topic_id', 'word', 'distribution'])

    def set_topic_docs(self):
        print('--- Set topic docs')
        all_topics = self.lda_model.get_document_topics(self.corpus, per_word_topics=False)
        all_topics_csr = gensim.matutils.corpus2csc(all_topics)

        all_topics_numpy = all_topics_csr.T.toarray()
        self.df_document_topic['topic_id'] = self.df_document_topic.apply(lambda row: all_topics_numpy[row.name].argmax(),
                                                                          axis=1)

    def build_score_lda_model(self, k):
        self.lda_model = gensim.models.LdaMulticore(corpus=self.corpus,
                                                    id2word=self.corpus_dict,
                                                    num_topics=k,
                                                    random_state=100,
                                                    chunksize=100,
                                                    passes=1,
                                                    workers=None,
                                                    per_word_topics=False)
        lda_model_coherence = CoherenceModel(model=self.lda_model, texts=self.corpus_doc_words,
                                             dictionary=self.corpus_dict, coherence='c_v')
        coherence_score = lda_model_coherence.get_coherence()

        for idx, topic in self.lda_model.show_topics(formatted=True, num_words=self.args.n_top_words):
            print('\n\nTopic: {} \nWords: {}'.format(idx, topic))

        print('# Topics: {}, Coherence score: {}'.format(str(k), str(coherence_score)))
        return coherence_score

    def topic_modelling(self):
        print('--- Topic modelling')
        custom_stop_words = ['19', 'covid', 'need', 'include', 'provide', 'numb', 'datum', 'set', 'result', 'value',
                             'different', 'sars', 'cov', 'coronavirus', 'mers', 'covid', 'il', 'increase', '10', 'ml',
                             'anti', 'high', 'day', 'increase', 'time', 'information', 'base', 'new', 'study',
                             'activity', 'report', 'group', 'year', 'test', 'case', 'level', 'ci', 'factor', 'ct', 'de',
                             '2009']

        self.df_document_topic = self.df[['hash_id', 'publish_time']].copy()

        self.df['text_to_model'] = self.df['processed_title'].fillna('') + ' ' + self.df['processed_abstract'].fillna('')

        text_to_model_as_list = self.df.text_to_model.values.tolist()
        self.corpus_doc_words = list(self.text_to_words_cleansed(text_to_model_as_list))
        self.corpus_doc_words = [[word for word in doc if word not in custom_stop_words] for doc in self.corpus_doc_words]
        self.corpus_dict = corpora.Dictionary(self.corpus_doc_words)
        self.corpus = [self.corpus_dict.doc2bow(text) for text in self.corpus_doc_words]

        if self.args.optimise:
            self.optimise_by_coherence()
        else:
            coherence_score = self.build_score_lda_model(k=self.args.n_topics)

        self.set_topic_docs()
        self.set_topic_words()

    def optimise_by_coherence(self):
        print('--- Optimise by coherence')
        best_score = 0
        best_topic_n = 0
        best_model_dir = 'step_dataprep_topic_modelling_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        best_model_file = 'best lda model.pkl'
        best_model_path = best_model_dir + '/' + best_model_file

        min_topics = 3
        max_topics = 12
        topics_range = range(min_topics, max_topics + 1)

        score_by_topics = {'n_topics': [],
                           'coherence_score': []}

        for k in topics_range:
            coherence_score = self.build_score_lda_model(k=k)
            score_by_topics['n_topics'].append(k)
            score_by_topics['coherence_score'].append(coherence_score)

            lda_vis = pyLDAvis.gensim_models.prepare(self.lda_model, self.corpus, self.corpus_dict)
            path = 'plots/Topic Modelling LDA Vis k_' + str(k) + '.html'
            pyLDAvis.save_html(lda_vis, path)
            if coherence_score > best_score:
                best_topic_n = k
                best_score = coherence_score
                print('Saving best LDA model with {} topics and cv score {}'.format(str(k), str(coherence_score)))
                os.makedirs(best_model_dir, exist_ok=True)
                self.lda_model.save(best_model_path)

        fig_name = 'LDA Topic Modelling Coherence By Topic'
        full_path = 'plots/' + fig_name + '.png'

        fig, ax = plt.subplots(figsize=(32, 14))
        sns.lineplot(data=score_by_topics, x='n_topics', y='coherence_score', linewidth=3)
        plt.xticks(list(topics_range))
        plt.ylabel('Coherence', fontsize=24)
        plt.xlabel('Topics', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.savefig(full_path, dpi=300, format='png')

        best_coherence_idx = score_by_topics['coherence_score'].index(max(score_by_topics['coherence_score']))
        best_n = score_by_topics['n_topics'][best_coherence_idx]
        print('{} topics achieved best coherence score of {}'.format(
            str(score_by_topics['n_topics'][best_coherence_idx]),
            str(score_by_topics['coherence_score'][best_coherence_idx])))

        if best_score != 0:
            print('Loading best model with {} topics'.format(str(best_topic_n)))
            self.lda_model = gensim.models.LdaMulticore.load(best_model_path)
            import shutil
            shutil.rmtree(best_model_dir)

    def get_topic_name(self, row):
        if row['topic_id'] == 0:
            return 'Clinical'
        if row['topic_id'] == 1:
            return 'Virology'
        if row['topic_id'] == 2:
            return 'Public Health'
        if row['topic_id'] == 3:
            return 'Genomics'
        if row['topic_id'] == 4:
            return 'Healthcare'

    def plot_topics_temporal_evolution(self):
        print('--- Plot topics temporal evolution')

        df_topic_cat = self.df_document_topic.copy()
        df_topic_cat['publish_year'] = df_topic_cat.publish_time.str[:4].fillna(-1).astype(int)
        df_topic_cat['topic_name'] = df_topic_cat.apply(lambda row: self.get_topic_name(row), axis=1)
        df_topic_cat = df_topic_cat.groupby(['publish_year', 'topic_name'])[['topic_name']].count().unstack().fillna(0)
        df_topic_cat = df_topic_cat.droplevel(axis=1, level=0).reset_index()
        df_topic_cat = pd.melt(df_topic_cat, id_vars=['publish_year'],
                               value_vars=['Clinical', 'Virology', 'Public Health', 'Genomics', 'Healthcare'])
        df_topic_cat.sort_values(by=['publish_year'], inplace=True)
        print(df_topic_cat)


        fig_name = 'Topics Temporal Evolution'
        fig, ax = plt.subplots(figsize=(26, 13))
        sns.lineplot(x='publish_year', y='value', hue='topic_name', linewidth=2, data=df_topic_cat)
        plt.xlabel('Publish Year', fontsize=24)
        plt.ylabel('Articles', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        leg = ax.legend()
        for line in leg.get_lines():
            line.set_linewidth(2.0)
        ax.legend(title='Topic', fontsize=20, title_fontsize=24)

        #self.run.log_image(fig_name, plot=fig)

        self.offline_save_fig(fig_name)

    def offline_save_fig(self, name):
        print('--- Offline save fig')
        if 'OfflineRun' in self.run._identity:
            full_path = 'plots/' + name + '.png'
            plt.savefig(full_path, dpi=300, format='png')
            print('Saved fig ', name)

    def output_datasets(self):
        print('--- Output Topic Datasets')
        if not (self.args.output_topics is None):
            os.makedirs(self.args.output_topics, exist_ok=True)
            path = self.args.output_topics + '/processed.csv'
            self.df_topics.to_csv(path, index=False)
            print('Output topics created: {}'.format(path))
            print('Column definition of topics output')
            print(self.df_topics.columns)

        if not (self.args.output_document_topic is None):
            os.makedirs(self.args.output_document_topic, exist_ok=True)
            path = self.args.output_document_topic + '/processed.csv'
            self.df_document_topic.to_csv(path, index=False)
            print('Output document topic created: {}'.format(path))
            print('Column definition of document topic output')
            print(self.df_document_topic.columns)


if __name__ == '__main__':
    print('--- Topic Modelling Started')
    topic_modelling = TopicModelling()
    print('--- Topic Modelling Completed')
