import os
import re
import sys
import traceback
from itertools import permutations

import numpy as np
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
from nltk import RegexpTokenizer
from numpy.random.mtrand import shuffle

from util.util import coherence, prep_text_to_stem


def to_log_file(string):
    file = open('logs.txt ', 'a', encoding='utf-8')
    file.write(string)

    file.close()


def _get_links(text):
    soup = BeautifulSoup(text, 'html.parser')

    return soup.find_all('a')


def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = np.unicode(text, 'utf-8')
    except (TypeError, NameError):  # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")

    return str(text)


def generate_pairs(bible_1, bible_2):

    temp = 0
    total = len(bible_1)
    pair_text = []
    for ref in bible_1.iterrows():

        sys.stdout.write('\r' + 'Loading: {:.2f}'.format((temp / total) * 100) + '%')
        sys.stdout.flush()
        temp += 1
        ref = ref[1]

        check = bible_2.loc[
            (bible_2['Book'] == ref['Book']) &
            (bible_2['Chapter'] == ref['Chapter']) &
            (bible_2['Verse'] == ref['Verse'])
            ]['Scripture'].empty

        if check is not True:
            verse_2 = bible_2.loc[
                (bible_2['Book'] == ref['Book']) &
                (bible_2['Chapter'] == ref['Chapter']) &
                (bible_2['Verse'] == ref['Verse'])
                ]['Scripture'].to_string(index=False)

            pair_text.append(ref['Scripture'] + '\t' + verse_2 + '\n')
    shuffle(pair_text)
    return pair_text


class TextPreprocess:
    datasets = None
    datasets_list = []
    root_dir = ''
    data_pairs = {}

    def __init__(self, dir_path):
        self.root_dir = dir_path
        self.datasets_list = os.listdir(dir_path)
        self.datasets = {}.fromkeys(self.datasets_list)
        pd.set_option('display.max_colwidth', -1)

    def set_root_dir(self, dir_path):
        self.root_dir = dir_path

    def get_prefix(self):
        return self.root_dir

    def get_dataset_list(self):
        return self.datasets_list

    def set_datasets(self, datasets):
        self.datasets = datasets

    def get_datasets(self):

        for name in self.datasets_list:

            path = self.root_dir + name

            try:
                self.datasets[name] = pd.read_csv(path, encoding='utf-8').drop_duplicates(subset='Scripture')

            except FileNotFoundError:
                return "The path " + path + " was not found."

        return self.datasets

    def clean_data(self, regex=None, auto_save=True):

        if regex is None:
            regex = []

        for name in self.datasets_list:
            print('\nCleaning: ', name)
            print('Progress: #', end='')
            path = self.root_dir + name

            try:
                dataset = pd.read_csv(path, encoding='utf-8')
            except FileNotFoundError:
                return "The path " + path + " was not found."

            temp = 0

            for exp in regex:
                sys.stdout.write('\r' + 'Loading: {:.2f}'.format(temp / len(regex) * 100) + '%')
                sys.stdout.flush()
                temp += 1
                dataset.replace(to_replace=exp, value=' ', regex=True, inplace=True)

            self.datasets[name] = dataset

            if auto_save is True:
                dataset.to_csv(path, index=False)

        return self.datasets

    def get_dataset(self, dataset_name):
        dataset = self.get_datasets()
        return dataset.get_value(dataset_name)

    def get_text_pairs(self):
        self.get_datasets()

        k_pairs = list(permutations(self.datasets.keys(), 2))

        print('\nCreating pairs: ')
        print('Progress: ', end='')
        progress = []
        for p in self.datasets.keys():
            progress.append('#')
            key = strip_accents(p.split(' ')[0]).lower()
            key += '-' + strip_accents('Português - Novo Testamento.csv'.split(' ')[0]).lower()
            print('\n\n' + p)
            print('Português - Novo Testamento.csv')
            pair_text = generate_pairs(self.datasets[p], self.datasets['Português - Novo Testamento.csv'])

            self.data_pairs[key] = pair_text

        return self.data_pairs

    def label_data(self, path):
        self.get_datasets()
        labels = []
        texts = []
        for key, data in zip(self.datasets.keys(), self.datasets.values()):

            for text in data['Scripture']:
                key = re.sub(r'\s-\sNovo\sTestamento.csv', '', key)
                labels.append(key)
                texts.append(text)

        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        df.to_csv(path, index=False)

    def stemming(self):

        dataset = self.get_datasets()
        data = dataset[0]
        scripture = data['Scripture']

        for verse in scripture:

            tokens = verse.split(r'\s')

            for token in tokens:
                letter = token.split()
                print(letter)
                break

    def save_pairs(self, file_type='.txt', dir_to_save=r'Resources/pairs/'):

        print('\nSaving pairs: ')
        print('Progress: #', end='')
        for key, text in zip(self.data_pairs.keys(), self.data_pairs.values()):

            path = dir_to_save + key + file_type
            print('#', end='')

            file = open(path, 'w', encoding='utf-8')

            for line in text:
                file.write(line)

            file.close()

    def collapse_verses(self, ref, verses_seq):

        for key, bible in zip(self.datasets.keys(), self.datasets.values()):

            script_seq = []

            for v_seq in verses_seq:
                check = bible.loc[
                    (bible['Book'] == ref['Book']) &
                    (bible['Chapter'] == ref['Chapter']) &
                    (bible['Verse'] == v_seq)
                    ]['Scripture'].empty

                verse_1 = bible.loc[
                    (bible['Book'] == ref['Book']) &
                    (bible['Chapter'] == ref['Chapter']) &
                    (bible['Verse'] == v_seq)
                    ]['Scripture'].to_string(index=False)

                if check is not True:
                    verse_1 = ' '.join(verse_1.split())
                    script_seq.append(verse_1)

            if len(script_seq) > 0:

                new_verse = ' '.join(script_seq)

                new_verse = re.sub(r'<sup>[(][0-9]*[-][0-9]*[)]<[/]sup>', '', new_verse)
                regexs = [r'[)]', r'[(]', r'[\[]', r'[\]]', r'[\{]', r'[\}]']
                to_str = [r'\)', r'\(', r'\]', r'\]', r'\{', r'\}']

                for res, to_str in zip(regexs, to_str):
                    script_seq[0] = re.sub(res, to_str, script_seq[0])

                try:

                    bible.replace(to_replace=script_seq[0], value=new_verse, regex=True, inplace=True)
                except re.error:
                    file = open('report/logs.txt', 'a', encoding='utf-8')
                    file.write(script_seq[0])
                    print(script_seq[0])
                    breakpoint()

                for v_seq in verses_seq[1:]:
                    try:
                        i = bible.loc[
                            (bible['Book'] == ref['Book']) &
                            (bible['Chapter'] == ref['Chapter']) &
                            (bible['Verse'] == v_seq)
                            ]['Scripture'].index.values.astype(int)
                        bible.drop(index=i, inplace=True)
                    except IndexError:
                        file = open('report/logs.txt', 'a', encoding='utf-8')
                        file.write(str(IndexError))
                        file.write(ref)
                        file.write(traceback.format_exc)
                        print(ref)
                        pass

                self.datasets[key] = bible

    def align_verses(self):
        global reference

        for k, b in zip(self.datasets.keys(), self.datasets.values()):
            temp = 0
            print('\n\nCollapsing verses : ', k, '...')

            total = len(b['Scripture'].index.values.astype(int))
            step = 1 / 100
            total_done = 0

            for index, reference in b.iterrows():

                sys.stdout.write('\r' + 'Loading: {:.2f}'.format((temp / total) * 100) + '%')
                sys.stdout.flush()
                temp += 1
                try:
                    search = BeautifulSoup(' '.join(reference['Scripture'].split(' ')), 'html.parser')

                    for tag in search.find_all('sup'):

                        search_1 = re.search(r'(?<=([(]))[0-9][0-9]*', tag.get_text())
                        if search_1 is not None:
                            first = search_1.group(0)

                        search_2 = re.search(r'(?<=([-]))[0-9][0-9]*', tag.get_text())

                        if search_2 is not None:
                            last = search_2.group(0)

                        if temp >= int(total * step):
                            total_done += temp
                            print('#', end='')

                        if search_1 is not None and search_2 is not None:
                            verses = np.arange(int(first), int(last) + 1)
                            self.collapse_verses(reference, verses)

                except TypeError:
                    string = k + '\n' + str(tag) + '\n' + str(reference) + traceback.format_exc()
                    to_log_file(string)
                except KeyError:
                    string = k + '\n' + str(tag) + '\n' + str(reference) + traceback.format_exc()
                    to_log_file(string)
                except AttributeError:
                    string = k + '\n' + str(tag) + '\n' + str(reference) + traceback.format_exc()
                    to_log_file(string)

        print('\nTotal done: {} Finished Successfully!'.format(total_done))


def stem_words(text, label):
    stem = AutoStem(text)
    stem.freq_counter()
    stem.stem_words()
    data = {label: list(filter(lambda x: type(x) == str, stem.select_stem()))}
    return data


def get_relative_count(count_dic):
    new_dict = {}
    total_count = sum(count_dic.values())
    for key, count in zip(count_dic.keys(), count_dic.values()):
        new_dict[key] = count / total_count
    try:
        new_dict.pop('')
    except:
        pass
    return new_dict


class AutoStem:
    path_dir = ''
    raw_text = ''
    candidates = None
    data = {
        'letter': {},
        'suffix': {},
    }
    top_suffixes = None
    suffixes_stem = None
    suffix_coh = None

    def __init__(self, text):

        """
        Initialize the required variables
        :param text: the text from where the stems will be extracted
        """
        self.candidates = {}
        self.suffixes_stem = {}
        self.suffix_coh = set()
        self.data['letter'] = {}
        self.data['suffix'] = {}

        self.raw_text = prep_text_to_stem(text)

    def get_suffix_freq(self):
        suffix_dic = self.data['suffix']
        return suffix_dic

    def get_letter_freq(self):
        letter_dic = self.data['letter']
        return letter_dic

    def get_text(self):
        return self.raw_text

    def freq_counter(self):

        tokens = self.raw_text.split()
        letters = []
        suffixes = []
        for token in tokens:
            letters.extend(token.strip())

            for size in range(1, 8):

                if len(token) > size:
                    suffix = token[-size:-1]
                    suffixes.append(suffix)

        suffixes.remove(sys.intern(''))
        self.data['letter'] = pd.Series(letters).value_counts(normalize=True).to_dict()
        self.data['suffix'] = pd.Series(suffixes).value_counts(normalize=True).to_dict()
        self.data['suffix'].pop('')

    def select_stem(self, threshold=100):
        """
        Selects the best stems
        Key args:

        threshold defines the number of suffix evaluated
        """

        selected = []
        top_suffixes = [tup[0] for tup in self.suffix_coh]

        for suffix in top_suffixes[:threshold]:
            stems = self.suffixes_stem[suffix]
            if len(set(stems)) >= 2:
                for stem in stems:
                    suffix = self.candidates[stem]
                    if len(set(suffix)) <= 5:
                        tokenizer = RegexpTokenizer(r'\w+', flags=re.UNICODE)
                        stem = ' '.join(tokenizer.tokenize(stem.lower()))
                        selected.append(stem)
        return selected

    def stem_words(self):
        """
         Stem the words based on the coherence
        """
        self.freq_counter()
        suffix_freq = self.data['suffix']

        suffixes = list(suffix_freq.keys())

        temp = 0

        total = len(suffixes)
        print('Loading: ', end='')

        print(suffixes[:10])
        for suffix in suffixes:
            self.suffixes_stem[suffix] = set()
            sys.stdout.write('\r' + 'Loading: {:.2f}'.format(temp / total * 100) + '%')
            sys.stdout.flush()

            search = re.findall(r'\w+' + str(suffix) + '#', self.raw_text)

            freq = suffix_freq[suffix]
            coh = coherence((suffix, freq), self.data['letter'])

            self.suffix_coh.add((suffix, coh))

            for word in set(search):
                stem = word.lower().replace(suffix + '#', '')
                self.suffixes_stem[suffix].add(stem)  # Saves the stem associated to the suffix

                try:
                    self.candidates[stem].add(suffix)
                except KeyError:
                    self.candidates.setdefault(stem, set(suffix))  # Saves the suffix associated with stem

            temp += 1

        self.suffix_coh = sorted(self.suffix_coh, key=lambda tup: tup[1], reverse=True)

    def get_data(self):
        return self.data

    def signatures(self):
        return self.candidates
