# Gensim in the name of the module refers to the Coherence Model from the gensim library
# pLSA itself is imported from a plsa module, because gensim does not provide pLSA

from plsa import Corpus, Pipeline, Visualize
from plsa.algorithms import PLSA
from plsa.preprocessors import tokenize
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as m_colors
import pandas as pd

from preprocessor import preprocess_normalized, preprocess_with_lemmatization


def build_plsa_coherence_graph(corpus, clean_text):
    # Testing to find the best number of topics
    plsa_coherence_results = []

    for number_of_topics in range(2, 13, 1):
        plsa = PLSA(corpus, number_of_topics, using_tf_idf)
        plsa_result = plsa.fit()

        result_for_coherence = []
        for topic in plsa_result.word_given_topic:
            result_for_coherence.append([])
            for word_tuple in topic:
                result_for_coherence[-1].append(word_tuple[0])

        coherence_model = CoherenceModel(topics=result_for_coherence, texts=clean_text,
                                         dictionary=Dictionary(clean_text), coherence='c_v')
        plsa_coherence_results.append(coherence_model.get_coherence())

    # Show graph
    x = range(2, 13, 1)
    plt.plot(x, plsa_coherence_results)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.savefig('plsa_coherence_measure_graph.png')


def perform_plsa(text, pipeline):
    corpus = Corpus(text, pipeline)

    plsa = PLSA(corpus, optimal_number_of_topics, using_tf_idf)
    result = plsa.fit()
    return result


def prepare_preprocessed_text(preprocessed_text):
    prepared_text = []
    for raw_text in preprocessed_text:
        prepared_text.append(" ".join(raw_text))
    return prepared_text


if __name__ == "__main__":
    # For the first time using: uncomment code for downloading ua corpus in the preprocessor module

    # Uncomment to use preprocessed text with lemmatization
    # text = preprocess_with_lemmatization()

    # Use preprocessed text without lemmatization
    text = preprocess_normalized()

    # # For that pLSA implementation, we will need to use each document not as a list, but as a string,
    # # so we apply additional preparation
    prepared_text = prepare_preprocessed_text(text)  # 142822 words
    #
    # # Performing pLSa
    optimal_number_of_topics = 7
    #
    using_tf_idf = True
    pipeline = Pipeline(tokenize)

    result = perform_plsa(prepared_text, pipeline)

    # Creating key-words list
    topics = []
    for topic in result.word_given_topic:
        topics.append([])
        topics[-1].append(len(topics) - 1)
        topics[-1].append([])
        topic = sorted(topic, key=lambda x: abs(x[1]))
        for word in topic[-10:]:
            topics[-1][-1].append(word)

    data_flat = [w for w_list in text for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=160)
    cols = [color for name, color in m_colors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.008)
        ax.set_ylim(0, 800)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords LDA', fontsize=22, y=1.05)
    plt.savefig('./resulting_plots/plsa/importance_of_topic_keywords_plsa.png')
