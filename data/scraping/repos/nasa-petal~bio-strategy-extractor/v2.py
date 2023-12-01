# ! pip install openai --quiet
# ! pip install -U pip setuptools wheel --quiet
# ! pip install -U spacy -- quiet
# ! python -m spacy download en_core_web_sm
# ! python3 -m pip install nltk

import openai
import spacy
import nltk
import gensim.downloader

from gensim.test.utils import common_texts
from gensim.models import Word2Vec, KeyedVectors
from textblob import TextBlob
from spacy import displacy
from openai.embeddings_utils import cosine_similarity

def main():
    nltk.download(['names',
                   'stopwords',
                   'state_union',
                   'twitter_samples',
                   'movie_reviews',
                   'averaged_perceptron_tagger',
                   'vader_lexicon',
                   'punkt',
                   'brown'])
    sp = spacy.load('en_core_web_sm')
    openai.api_key = 'OPENAI API KEY'

    # experiment with word2vec embedding similarity
    glove_vector = gensim.downloader.load('glove-wiki-gigaword-100')
    glove_vector.most_similar("man-made")

    glove_vector = gensim.downloader.load('fasttext-wiki-news-subwords-300')
    glove_vector.most_similar("man-made")

    # word2vec analogy feature! (try with king - man + woman, should generate queen)
    glove_vector.most_similar(negative = ["man-made"],
                              positive = ["bicyclist", "natural"])[0][0]

    # experiment with openai embedding similarity
    synthetic = openai.Embedding.create(input = "man-made",
                                        model = 'text-embedding-ada-002')['data'][0]['embedding']
    natural = openai.Embedding.create(input = "nature-made", 
                                      model = 'text-embedding-ada-002')['data'][0]['embedding']
    cyclist = openai.Embedding.create(input = "cyclist", 
                                      model = 'text-embedding-ada-002')['data'][0]['embedding']
    syn_distance = cosine_similarity(synthetic, cyclist)
    nat_distance = cosine_similarity(natural, cyclist)
    print(syn_distance, nat_distance)

    # find most generalizable term
    # search for similar words
    # (uses a theory i came up with that the more often 
    # a word comes up in the corpus, the more generalizable it is)
    function = ["reduce", "excess", "water", "runoff"]
    word_counts = {glove_vector.vocab[token].count : token for token in function}
    min_word = word_counts[min(list(word_counts.keys()))]
    glove_vector.most_similar(min_word)

    # extract the functions and context from the biomimicry toolbox website
    function, contexts = extract("How might we make urban cyclists on the street more visible to drivers at night?")
    print(function, contexts)
    function, contexts = extract("How might we reduce the use of toxic substances in paints?")
    print(function, contexts)
    function, contexts = extract("How can one design a container to retain liquids efficiently?")
    print(function, contexts)
    function, contexts = extract("How might we keep buildings cool in the summer?")
    print(function, contexts)
    function, contexts = extract("How might we reduce excess water runoff during rainstorms in cities?")
    print(function, contexts)

def extract(engr_question):
  doc = sp(engr_question)
  noun_phrases = TextBlob(engr_question).noun_phrases
  phrases_to_ignore = {}                                                                                                                      # filter out all extraneous information
  for token in doc:
    if token.pos_ == "PUNCT":
      phrases_to_ignore[engr_question.index(token.orth_)] = token.orth_
    if token.dep_ == "nsubj":
      found_in_noun_phrase = False
      for noun_phrase in noun_phrases:
        if token.orth_ in noun_phrase:
          index = engr_question.index(token.orth_) - noun_phrase.index(token.orth_)
          phrases_to_ignore[index] = noun_phrase
          found_in_noun_phrase = True
      if not found_in_noun_phrase:
        phrases_to_ignore[engr_question.index(token.orth_)] = token.orth_
  phrases_to_ignore[0] = engr_question[0:list(phrases_to_ignore.keys())[0] - 1]

  function_in_context = ""
  index = 0
  while(index < len(engr_question)):
    curr_char = engr_question[index]
    if index in list(phrases_to_ignore.keys()):
      index += len(phrases_to_ignore[index]) + 1
    else:
      function_in_context += curr_char
      index += 1

  doc = sp(function_in_context)                                                                                                               # find the contexts
  contexts = [' '.join([subtoken.orth_ for subtoken in token.subtree]) for token in doc if token.pos_ == "ADP"]
  for context in contexts:
    if "of" in context:
      contexts.remove(context)

  function = ""                                                                                                                               # find the function
  for token in doc:                                                                                                                           # fix this (for when there are multiple occurrences of the same context)
    token_found = False
    for context in contexts:
      if token.orth_ in context:
        token_found = True
    if not token_found:
      function += token.orth_ + " "
  function = function[:-1]

  return function, contexts

if __name__ == "__main__":
    main()
