import argparse
import os
import re
import sys
from glob import glob

import gensim
import pandas as pd
import spacy
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel, LdaModel
from gensim.test.utils import common_texts, datapath
from gensim.utils import simple_preprocess


def preProcess(text):
    # Lemmatization
    sp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    text = re.sub(r'[,\.!?]', '', text)
    text = text.lower()

    words = gensim.utils.simple_preprocess(text, deacc=True)  # Tokenize

    data = [words]

    # Build the bigram model
    bi = gensim.models.Phrases(data, min_count=5, threshold=100)
    bi_mod = gensim.models.phrases.Phraser(bi)

    # Remove stopwords
    sWords = sp.Defaults.stop_words

    lim = len(words)
    i = 0
    for w in words:
        if words[i] in sWords:
            words.pop(i)
            lim = len(words)

        i += 1
        if i >= lim:
            break

    word_bigrams = [bi_mod[data[0]]]

    lemmatizedText = []

    spacyLemmas = sp(" ".join(word_bigrams[0]))
    permit = ['NOUN', 'ADJ', 'VERB', 'ADV']

    for t in spacyLemmas:
        if t.pos_ in permit:
            lemmatizedText.append(t.lemma_)

    return [lemmatizedText]


def calcCoherence(lemmatizedTexts, passes=100, nTopics=5, workers = 1):

    id2word = Dictionary(lemmatizedTexts)
    corp = [id2word.doc2bow(text) for text in lemmatizedTexts]

    ldaModel = gensim.models.LdaMulticore(
        corpus=corp,
        id2word=id2word,
        num_topics=nTopics,
        passes=passes,
        random_state=100,
        per_word_topics=False,
        alpha=0.01,
        eta=0.9,
        workers=workers
    )

    coherenceModel = CoherenceModel(
        model=ldaModel, texts=lemmatizedTexts, dictionary=id2word, coherence='c_v', processes=0
    )

    return coherenceModel.get_coherence()


def outputScoreData(score, docName, outFile):
    df = pd.DataFrame([[docName, "lda_coherence", score]], columns=["filename", "measure", "score"])

    if outFile != "":
        with open(outFile, "w") as fh:
            fh.write(df.to_csv(sep='\t', index=False))


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_files_pattern',
    help='the glob pattern for the input TXT files to calculate coherence for',
    required=True,
)
parser.add_argument(
    '--gcdc', action='store_true', help='flag to indicate the input files are GCDC CSV files'
)
parser.add_argument('--out_file', help='the output file for csv results', required=True)
parser.add_argument('--passes', help='number of passes over texts', default=100, type=int)
parser.add_argument('--topics', help='number of topics to consider', default=5, type=int)
parser.add_argument('--workers', help='number of workers threads to use', default=1, type=int)

args = parser.parse_args()

input_files = glob(args.input_files_pattern)
if not input_files:
    raise FileNotFoundError(f'no files found for pattern: {args.input_files_pattern}')
scores = []


def calculate_coherence_score(text) -> float:
    lemmatizedText = preProcess(text)
    ldaCoherence = calcCoherence(lemmatizedText, args.passes, args.topics, args.workers)
    return ldaCoherence


if not args.gcdc:
    for text_file in input_files:
        batch_id = text_file.split('/')[-2]
        try:
            print('reading:', text_file)
            with open(text_file, "r", encoding='utf-8') as pFile:
                text = pFile.read()
                scores.append(
                    (
                        batch_id,
                        os.path.basename(text_file),
                        "lda_coherence",
                        calculate_coherence_score(text),
                    )
                )
        except Exception as err:
            print(err, file=sys.stderr)

    df = pd.DataFrame(scores, columns=["section", "filename", "measure", "score"])
else:
    df = None

    for csv_file in input_files:
        try:
            print('reading:', csv_file)
            curr_df = pd.read_csv(csv_file)
            curr_df['measure'] = 'lda_coherence'
            curr_df['filename'] = os.path.basename(csv_file)
            curr_df['score'] = curr_df['text'].apply(calculate_coherence_score)
            curr_df['text_size'] = curr_df['text'].apply(lambda x: len(x))
            curr_df = curr_df.drop(columns='text')
            if df is None:
                df = curr_df
            else:
                df = pd.concat([df, curr_df])
        except Exception as err:
            print(err, file=sys.stderr)

print('writing:', args.out_file)
df.to_csv(args.out_file, index=False)
