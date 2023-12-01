import os
import joblib
import json
import argparse
import numpy as np
from gensim.models import CoherenceModel

MAX_WORDS_PER_TOPIC = 10
START_COMMENT = "% Inicio de tabela\n\n"
END_COMMENT = "\n\n% Fim de tabela\n\n"

TABLE_TEMPLATE = """
\\begin{table}[ht]
\centering
\\footnotesize
\caption{$TABLE_CAPTION}
\label{$TABLE_LABEL}
\\begin{tabular}{| c | c | c |}
\hline
tópico & palavras & coerência \\\\ \hline
$TABLE_LINES
\end{tabular}
\end{table}
"""

def get_coherence_score_for_each_topic(topics, documents, dictionary, coherence="c_npmi", no_of_words=20):
    """Calculates topic coherence using gensim's coherence pipeline.

    Parameters:

    topics (list of str list): topic words for each topic
    
    documents (list of str): set of documents

    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset

    coherence (str): coherence type. Can be 'c_v', 'u_mass', 'c_uci' or 'c_npmi'

    Returns:

    float: coherence score
    """
    coherence_model = CoherenceModel(
                topics=topics, 
                texts=documents, 
                dictionary=dictionary, 
                coherence=coherence,
                processes=0,
                topn=no_of_words
    )

    return coherence_model.get_coherence_per_topic()

parser = argparse.ArgumentParser(description='Topic tables generator')
parser.add_argument('--models', nargs='+', help='list of model files', required=True)
parser.add_argument('--dictionary', type=str, help='model file', required=True)
parser.add_argument('--test_docs', type=str, help='model file', required=True)
args = parser.parse_args()


def get_model_name(model_path):
    return model_path.split(os.path.sep)[-1]


def get_model_type(model_path):
    model_name = get_model_name(model_path)
    return model_name.split('_')[0].upper()


def get_topic_line(idx, topic, coherence_by_topic):
    topic_str = ' '.join(topic[:MAX_WORDS_PER_TOPIC])
    return f'\(T_{{{idx}}}\) & {topic_str} & {round(coherence_by_topic[idx], 6)} \\\\ \hline'.replace(".", ",")


def get_model_table(model_path, test_docs, dictionary):
    model = joblib.load(model_path)
    coherence_by_topic = np.array(get_coherence_score_for_each_topic(model["topics"], test_docs, dictionary))
    sorted_indices = coherence_by_topic.argsort()[::-1]

    model_table = TABLE_TEMPLATE
    
    model_type = get_model_type(model_path)
    lines = []
    for idx in sorted_indices:
        lines.append(get_topic_line(idx, model["topics"][idx], coherence_by_topic))
    
    table_lines = '\n'.join(lines)
    table_label = f'tab:modelo{model_type.lower().capitalize()}'
    table_caption = f'Tópicos do modelo {model_type} com \(k={len(model["topics"])}\) em ordem decrescente de coerência.'
    model_table = model_table.replace("$TABLE_LINES", table_lines)
    model_table = model_table.replace("$TABLE_LABEL", table_label)
    model_table = model_table.replace("$TABLE_CAPTION", table_caption)

    return model_table


test_docs = json.load(open(args.test_docs, "r"))["split"]
dictionary = joblib.load(args.dictionary)

tables = ""

for model_path in args.models:
    model_table = get_model_table(model_path, test_docs, dictionary)
    tables += START_COMMENT + model_table + END_COMMENT

print(tables)