from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import pandas as pd

PROPOSALS_PATH = './data/proposals_clean.csv'

proposals_df = pd.read_csv(PROPOSALS_PATH, ';')

# Concatenate string values of multiple columns
def combine(row, columns):
    items = [row[column] for column in columns]
    items = filter(lambda x: isinstance(x, str) and len(x) > 0, items)
    return " ".join(items)

proposals_df['document'] = proposals_df.apply(lambda x: combine(x, proposals_df.columns[-8:]), axis=1)

documents = proposals_df.document.str.split().to_numpy()

# Prepare corpus for latent semantic analysis
def prepare_corpus(documents):

  print('Preparing corpus...')

  dictionary = corpora.Dictionary(documents)
  document_term_matrix = [dictionary.doc2bow(document) for document in documents]

  print('Done!')

  return dictionary, document_term_matrix

# Generate LSA model for cleaned, tokenised documents using the given number of topics
def generate_lsa_model(documents, n_topics):

  dictionary, document_term_matrix = prepare_corpus(documents)

  print('Generating the LSA model...')

  lsaModel = LsiModel(document_term_matrix, id2word=dictionary, num_topics=n_topics)
  print(lsaModel.print_topics(num_topics=n_topics, num_words=10))
  print('Done!')

  return lsaModel

def compute_coherence_values(dictionary, document_term_matrix, documents, stop, start=2, step=3):

  print(f'Computing coherence values starting from {start}, stopping at {stop}, using step {step}')

  coherence_values = []
  for n_topics in range(start, stop, step):
      print('Current number of topics: ' + str(n_topics))
      model = LsiModel(document_term_matrix, num_topics=n_topics, id2word = dictionary)

      coherencemodel = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
      coherence_values.append(coherencemodel.get_coherence())

  print('Done!')

  return coherence_values

def plot_coherence_values(documents, start, stop, step):

  dictionary, document_term_matrix = prepare_corpus(documents)
  coherence_values = compute_coherence_values(dictionary, document_term_matrix, documents, stop, start, step)

  print('Plotting coherence values...')

  x = range(start, stop, step)
  plt.plot(x, coherence_values)
  plt.xlabel('Number of Topics')
  plt.ylabel('Coherence score')
  plt.legend(('coherence_values'), loc='best')
  plt.savefig('./plots/coherence_values.png')

  print('Plot saved!')

generate_lsa_model(documents, 100)