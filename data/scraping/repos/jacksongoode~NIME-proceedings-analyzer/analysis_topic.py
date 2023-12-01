# This file is part of the NIME Proceedings Analyzer (NIME PA)
# Copyright (C) 2022 Jackson Goode, Stefano Fasciani

# The NIME PA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The NIME PA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# If you use the NIME Proceedings Analyzer or any part of it in any program or
# publication, please acknowledge its authors by adding a reference to:

# J. Goode, S. Fasciani, A Toolkit for the Analysis of the NIME Proceedings
# Archive, in 2022 International Conference on New Interfaces for
# Musical Expression, Auckland, New Zealand, 2022.

# Native
import sys
if sys.version_info < (3, 7):
    print("Please upgrade Python to version 3.7.0 or higher")
    sys.exit()
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
import imp
import os
from os import path
import io
import re
import pickle
import collections
import argparse
from datetime import datetime

# External
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaMulticore, LdaModel
import pyLDAvis, pyLDAvis.gensim_models
import pandas as pd
import numpy as np

import nltk
nltk.download('punkt', download_dir='./cache/nltk_data', quiet=True)
nltk.download('wordnet', download_dir='./cache/nltk_data', quiet=True)
nltk.download('omw-1.4', download_dir='./cache/nltk_data', quiet=True)
nltk.data.path.append('./cache/nltk_data/')

# Helper
import pa_print
from pa_extract import clean_text
from pa_utils import import_config, boolify
from pa_load import load_bibtex, extract_bibtex
from tqdm import tqdm

# Variables
lda = LdaMulticore
grobid_text_src = './cache/text/grobid/'
lda_src = './cache/lda/'

def gen_model(remodel=True, rebuild=True, model='', num_topics=5, user_config=None):
    # * Load model
    if path.isfile(lda_src+model) and not (remodel or rebuild):
        pa_print.nprint('\nLoading bodies, dict, corpus, and model...')
        processed_bodies = pickle.load(open(f'{lda_src}bodies.pkl', 'rb'))
        dictionary = gensim.corpora.Dictionary.load(f'{lda_src}dictionary.gensim')
        corpus = pickle.load(open(f'{lda_src}corpus.pkl', 'rb'))
        lda_model = lda.load(f'{lda_src}{model}')

    else: # Build model afterwards
        # Load resources
        if path.isfile(f'{lda_src}dictionary.gensim') and path.isfile(f'{lda_src}corpus.pkl') and not rebuild:
            pa_print.nprint('\nLoading bodies, dict and corpus...')
            processed_bodies = pickle.load(open(f'{lda_src}bodies.pkl', 'rb'))
            dictionary = gensim.corpora.Dictionary.load(f'{lda_src}dictionary.gensim')
            corpus = pickle.load(open(f'{lda_src}corpus.pkl', 'rb'))
        else:
            # Remove old
            for doc in [f'{lda_src}bodies.pkl', f'{lda_src}dictionary.gensim', f'{lda_src}corpus.pkl']:
                try: os.remove(doc)
                except FileNotFoundError: pass

            # Build everything from text files
            pa_print.nprint('Building dict and corpus...')
            doc_list = []
            processed_bodies = []

            for text_fn in os.listdir(grobid_text_src):
                if text_fn.startswith('grob_'):
                    with open(grobid_text_src+text_fn, 'r') as doc:
                        doc_list.append(doc.read())

            for doc in doc_list:
                processed_words = clean_text(doc, user_config) # extract only meaningful words, user config!
                processed_bodies.append(processed_words)

            # Save processed bodies for coherence score
            pickle.dump(processed_bodies, open(f'{lda_src}bodies.pkl', 'wb'))

            # Make and save dict and corpus
            dictionary = corpora.Dictionary(processed_bodies)
            dictionary.filter_extremes(no_below=3) # remove those with counts fewer than 3
            dictionary.save(f'{lda_src}dictionary.gensim')

            corpus = [dictionary.doc2bow(doc) for doc in processed_bodies]
            pickle.dump(corpus, open(f'{lda_src}corpus.pkl', 'wb'))

        # Build LDA model - default settings
        if remodel or rebuild or not path.isfile(f'{lda_src}{model}'):
            pa_print.nprint('Building model...')
            alpha ='asymmetric'
            eta = 0.5
            lda_model = lda(corpus, num_topics=num_topics, id2word=dictionary,
                            random_state=100, passes=10, alpha=alpha, eta=eta, per_word_topics=True)
            date = datetime.now().strftime('%Y%m%d')
            lda_model.save(f'{lda_src}{date}-{num_topics}-{alpha}-{eta}.model')
            pa_print.nprint('Saved model!')
        else: lda_model = lda.load(f'{lda_src}{model}')

    return processed_bodies, dictionary, corpus, lda_model

def gen_titles(user_config):
    bib_db = load_bibtex('./cache/bibtex/nime_papers.bib')
    bib_db = extract_bibtex(bib_db, args)
    processed_titles = []

    for pub in bib_db:
        title = clean_text(pub['title'], user_config)
        processed_titles.append(title)

    return processed_titles

def gen_lda(lda_model, corpus, processed_bodies, dictionary):
    # Compute Perplexity
    pa_print.nprint(f'Perplexity: {lda_model.log_perplexity(corpus)}')  # a measure of how good the model is, lower the better

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_bodies, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    pa_print.nprint(f'Coherence Score: {coherence_lda}')

    # Show some visualization of the topics that gathered
    lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_display, './output/lda.html')
    pa_print.nprint('Generated lda.html in ./output!')

def gen_wordcloud(processed_data):
    from wordcloud import WordCloud

    for data in processed_data:
        words = [word for doc in data[1] for word in doc]
        counter = dict(collections.Counter(words))
        wc = WordCloud(width=1920, height=1444,
                        background_color="white", max_words=500
                        ).generate_from_frequencies(counter)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(f'./output/wordcloud_{data[0]}.png', dpi=300)
    pa_print.nprint('Generated .png files in ./output!')

def gen_topic_plots(corpus, lda_model, year_dict, year_list, year_start, year_end):
    year_counts = np.zeros(year_end-year_start)

    # Add topic distribution from each doc into buckets of years
    for i in range(len(corpus)):
        topics = lda_model.get_document_topics(corpus[i])
        for j in range(year_start, year_end):
            if year_list[i][0] == j:
                year_counts[j-year_start] += 1 # how many bodies in each year
                for k, year_top in enumerate(year_dict[j]):
                    for top in topics:
                        if str(year_top[0]) == str(top[0]):
                            year_top = list(year_top)
                            year_top[1] = float(year_top[1]) + float(top[1])
                            year_dict[j][k] = tuple(year_top)

    # Weight the topic values by numbers of papers published each year
    for key, val in year_dict.items():
        for index, j in enumerate(val):
            j = list(j)
            j[1] = float(j[1]) / year_counts[index]
            year_dict[key][index] = tuple(j)

    # Create empty dict of lists for year range (n topics each year)
    xvals = [ [] for _ in range(num_topics) ]
    yvals = [ [] for _ in range(num_topics) ]
    plt.figure(figsize=(20,10))

    for year, topics in year_dict.items():
        for topic in topics:
            xvals[topic[0]].append(int(year))
            yvals[topic[0]].append(topic[1])

    for i in range(num_topics):
        plt.scatter(xvals[i], yvals[i], label=f'Topic {i}')
        s = UnivariateSpline(xvals[i], yvals[i], s=.1)
        xs = np.linspace(year_start, year_end, 50)
        ys = s(xs)
        plt.plot(xs, ys, label=f'Spline for topic {i}')

    plt.legend()
    plt.ylim(bottom=0)
    plt.xticks(range(year_start, year_end))
    plt.xlabel('Year')
    plt.ylabel('Occurrence of Topic over Yearly Papers)')
    plt.title('Occurrence of Topics over Publication Year')
    plt.savefig('./output/topic_occurrence.png')

    pa_print.nprint('Generated diagram .png in ./output!')

def gen_counts(processed_data, year_list):
    top_counts_dfs = {}
    alt_top_counts_dfs = {}
    unique_dfs = {}
    abs_unique_dfs = {}

    for data in processed_data:
        # * Most popular keywords for each year (100)
        yearly_bodies, top_counts = {}, {}

        for year, doc in zip(year_list, data[1]):
            year = year[0]
            try:
                yearly_bodies[year].extend(doc) # accum all words from each year's papers
            except:
                yearly_bodies[year] = []
                yearly_bodies[year].extend(doc)

        for year in yearly_bodies:
            counts = collections.Counter(yearly_bodies[year])
            top_counts[year] = counts.most_common(100) # take most common

        top_counts = collections.OrderedDict(sorted(top_counts.items()))

        # Two columns [year, ('term', count)] - for Google Sheets
        top_counts_df = pd.DataFrame([[i,j] for i in top_counts.keys() for j in dict(top_counts[i]).items()])
        top_counts_dfs[data[0]] = top_counts_df

        # Columns by years (20 columns)
        alt_top_counts_df = pd.DataFrame.from_dict(top_counts, orient='index')
        alt_top_counts_dfs[data[0]] = alt_top_counts_df

        # * Get unique counts by removing last years top 10 (looking backwards)
        unique_counts = {}
        old_top, old_years = [], []

        for i, year in enumerate(top_counts):
            cur_counts = dict(top_counts[year]) # keep a dict for counts
            # cur_words = list(cur_counts) # unpack keys into list

            # new dict, without past year
            old_years.append(year)

            # remove words from prior years
            for key in old_top:
                cur_counts.pop(key, None)

            unique_words = list(dict(cur_counts))[:5] # make list of top 5 words
            old_top.extend(unique_words) # add old top to del words

            unique_counts[year] = cur_counts.items() # reassign
            # pa_print.nprint(unique_words)

        unique_df = pd.DataFrame.from_dict(unique_counts, orient='index')
        unique_dfs[data[0]] = unique_df

        # * Get absolute unique terms per year (not in the top common words of all other years)
        # Similar process to above but looks both forward and backward
        abs_unique_counts = {}

        for i, year in enumerate(top_counts):
            cur_counts = dict(top_counts[year]) # keep a dict for counts
            cur_words = list(cur_counts) # unpack keys into list (for a set)

            # new dict, without current year
            later_counts = {x: top_counts[x] for x in top_counts if x != year}

            other_words = []
            for later_year in later_counts:
                later_words = list(dict(later_counts[later_year]))
                other_words.extend(later_words) # extend

            unique_words = set(cur_words) - set(other_words)
            del_words = set(cur_words) - set(unique_words)

            for key in del_words: # del words included other years common words
                cur_counts.pop(key)
            abs_unique_counts[year] = list(cur_counts.items())

        abs_unique_df = pd.DataFrame.from_dict(abs_unique_counts, orient='index')
        abs_unique_dfs[data[0]] = abs_unique_df

    with pd.ExcelWriter('./output/topics.xlsx') as writer:
        for name in ['bodies', 'titles']:
            top_counts_dfs[name].to_excel(writer, sheet_name=f'Top counts {name}', header=False)
            alt_top_counts_dfs[name].to_excel(writer, sheet_name=f'Alt top counts {name}', header=False)
            unique_dfs[name].to_excel(writer, sheet_name=f'Unique counts {name}', header=False)
            abs_unique_dfs[name].to_excel(writer, sheet_name=f'Absolute unique counts {name}', header=False)

        topic_row = pd.Series(data=lda_model.show_topics(num_words=10), name='Word constituents of topics')
        topics_df = pd.DataFrame.from_dict(year_dict, orient='index')
        topics_df = topics_df.append(topic_row, ignore_index=False)
        topics_df.to_excel(writer, sheet_name='Weighted topics')

    pa_print.nprint('\nGenerated topics.xlsx in ./output!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze a publication given a BibTeX and directory of pdf documents')
    parser.add_argument('-n', '--nime', action='store_true', default=False,
                        help='uses NIME based corrections')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='prints out analysis process and results')
    args = parser.parse_args()

    # Sets global print command
    pa_print.init(args)

    # Print notice
    pa_print.lprint()

    # User config
    user_config = import_config('./resources/custom.csv')
    selected_years = user_config[3]
    if len(selected_years) != 0: # years
        int_years = list(map(int,selected_years))
        year_start, year_end = min(int_years), max(int_years)+1
    else:
        year_start, year_end = 2001, 2021

    # Make sure dirs exist
    for d in [lda_src,'./output']:
        os.makedirs(d, exist_ok=True)

    # Question for load dict, corpus, model for docs
    remodel, rebuild = True, True
    model = ''
    answer = int(input('\nWant to [1] remodel, [2] rebuild dictionary and corpus, [3] both, or [4] load model? (1,2,3,4): '))
    if answer == 1:
        rebuild = False
        num_topics = int(input('Number of topics?: '))
    elif answer == 2:
        remodel = False
    elif answer == 3:
        num_topics = int(input('Number of topics?: '))
    elif answer == 4:
        rebuild, remodel = False, False
        pa_print.nprint('\nWhich model?')
        models = [mod for mod in os.listdir(lda_src) if mod.endswith('.model')]
        for i, mod in enumerate(models):
            print(f'{i+1}: {mod}')
        answer = int(input('\nSelect an option: ')) - 1
        model = models[answer]
        num_topics = int(model.split('-')[1])

    # Create list to mark each text with year (will be linked to corpus values)
    year_list = []
    for i in os.listdir(grobid_text_src):
        if i.startswith('grob_'):
            name = i.split('grob_nime')[-1]
            year = name.split('_')[0]
            year_list.append((int(year), name))

    # Create empty dict of lists for years (n topics each year)
    year_dict = dict()
    for i in range(year_start, year_end):
        year_dict[i] = []
        for j in range (0, num_topics):
            year_dict[i].append((j, 0))

    processed_bodies, dictionary, corpus, lda_model = gen_model(remodel, rebuild, model, num_topics, user_config)

    # Use titles for processed words
    processed_titles = gen_titles(user_config)

    processed_data = [('bodies', processed_bodies), ('titles', processed_titles)]

    # * LDA
    answer = boolify(input('\nGenerate LDA scores & visualizations? (y/N): '))
    if answer:
        gen_lda(lda_model, corpus, processed_bodies, dictionary)

    # * Wordcloud
    answer = boolify(input('\nGenerate wordcloud diagrams? (y/N): '))
    if answer:
        gen_wordcloud(processed_data)

    # * Plot topics
    answer = boolify(input('\nGenerate topic plots? (y/N): '))
    if answer:
        gen_topic_plots(corpus, lda_model, year_dict, year_list, year_start, year_end)

    # * Counts
    answer = boolify(input('\nGenerate top and unique counts? (y/N): '))
    if answer:
        gen_counts(processed_data, year_list)
