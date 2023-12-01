# Libraries
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize as wt

# Read the preprocessed reviews
preprocessed_reviews = pd.read_csv('preprocessed_reviews.csv')

sample_review = preprocessed_reviews['review']


# Tokenize the reviews
reviews = sample_review.apply(wt)

# Topic Modeling
# Latent Dirichlet Allocation (LDA)
# This creates a mapping from words to their integer IDs
dictionary = Dictionary(reviews)

# creates a "bag-of-words" representation for a review. It returns a list of (word_id, word_frequency) tuples.
corpus = [dictionary.doc2bow(review) for review in reviews]

lda_model = LdaModel(
    # The bag-of-words representation of the reviews.
    corpus=corpus,
    # This parameter represents the document-topic density. With a higher alpha,
    # documents are likely to be composed of more topics,
    alpha=0.1,
    # This parameter represents the topic-word density. With a higher eta, topics are likely to be composed of more words (terms),
    # resulting in a denser topic-word distribution.
    eta='symmetric',
    # A mapping from word IDs to words, which helps interpret the topics.
    id2word=dictionary,
    # The number of topics the model should discover.
    num_topics=3,
    # This ensures reproducibility (the seed)
    random_state=42,
    # The number of times the algorithm should traverse the corpus
    passes=25,
    # Per-word topic assignments should be computed, not just per-document topic distributions
    per_word_topics=False
)

# Print the topics. -1 - all topics will be in result
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")


# Prepare the LDA model visualization
lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)

# Display the visualization
pyLDAvis.display(lda_vis)

# Save the visualization
pyLDAvis.save_html(lda_vis, 'lda.html')

# Calculate coherence score (c_v only)
coherence_model_lda = CoherenceModel(
    model=lda_model,
    texts=reviews,
    dictionary=dictionary,
    coherence='c_v'
)

coherence_lda = coherence_model_lda.get_coherence()
print(f"Coherence score: {coherence_lda}")
