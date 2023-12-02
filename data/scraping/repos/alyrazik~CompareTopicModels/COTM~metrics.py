import matplotlib.pyplot as plt
import numpy as np
from gensim.models.coherencemodel import CoherenceModel



def diversity(tokens):
    """It defines topic diversity as the percentage of the unique words in the top 25 words
of all topics
Args:
    tokens: list of tokens
returns:
    diversity score as a percentage of unique words in top 25 tokens.
    """
    return len(set(tokens)) / len(tokens)


def show_progress(metrics):
    """ charts the metrics that were logged during model training.
    Args:
        metrics: dictionary with keys of metric names and values are lists of measured values.
    returns:
        displays the figure with axes to display trends.
        """

    fig = plt.figure(figsize=(10, 10))
    length_ = len(metrics)
    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        plt.subplot(length_ // 2, length_ - length_ // 2, i+1)
        plt.errorbar(x=np.arange(len(metric_values)),
                     y=metric_values
                     )
        # plt.title(f"{metric_name}")
        plt.xlabel('Pass number')
        plt.ylabel(f"{metric_name}")
    plt.show()
    return fig



def assess_model(model, corpus, dictionary, n_topics=None):
    topics = []
    metrics = {}
    for i in range(n_topics):
        topic = model.show_topic(i, 12)
        topics.append([token for (token, probability) in topic])

    # Log the metrics
    # coherence
    cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    metrics['coherence'] = cm.get_coherence()
    # diversity
    tokens = []  # To calculate diversity, obtain the most probable 25 tokens across all topics.
    for i in range(n_topics):
        for item in model.show_topic(i, topn=50):  # 50 is chosen heuristically to include most probable tokens.
            tokens.append(item)
    # print(tokens)
    sorted_tokens = sorted(tokens, key=lambda x: x[1], reverse=True)
    # print(sorted_tokens)
    metrics['diversity'] = diversity([token for (token, prob) in sorted_tokens][:25])
    # # perplexity
    # metrics['perplexity'] = model.log_perplexity(list(corpus))
    return metrics
