"""
FILE TO BE RUN FIRST BEFORE CAPEC LDA AND LSA FILES
Preprocess document to be fed into topic models.

"""
import pickle

if __name__ == '__main__':
    import string
    import csv
    from collections import defaultdict
    import re
    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel

    # spacy for lemmatization
    import spacy
    # Data cleaning
    import re
    from nltk.corpus import stopwords
    stopWords = stopwords.words('english')
    stopWords.extend(["from", "subject", "re", "edu", "use"])
    nlp = spacy.load("en_core_web_sm", disable=["parser,ner"])
    columns = defaultdict(list)
    rows = defaultdict(list)


    def sent_to_words(sentences):
        count = 0
        for sentence in sentences:
            count += 1
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


    def remove_stopwords(text):
        return [[word for word in simple_preprocess(str(doc)) if word not in stopWords] for doc in text]


    def removepunctuation(supplied_text):
        translator = str.maketrans('', '', string.punctuation)
        return supplied_text.translate(translator)


    def removewhitespaces(supplied_text):
        return " ".join(supplied_text.split())

    def lemmatization(texts, allowed_pos=["NOUN", "ADJ", "VERB", "ADV"]):
        count = 0
        text_out = list()
        for sent in texts:
            doc = nlp(" ".join(sent))
            text_out.append([token.lemma_ for token in doc if token.pos_ in allowed_pos])
        return text_out


    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]


    def list_to_string(text):
        filtered_text_string = ""
        for ele in text:
            filtered_text_string += ele + " "
        return filtered_text_string


    id2word = corpora.Dictionary()
    result = []

    with open('CAPEC/Comprehensive CAPEC Dictionary.csv', encoding="utf8") as csvfile:

        reader = csv.DictReader(csvfile)
        capec_entry = 1
        for row in reader:
            ap_text = ""
            for val in row.values():
                obj = val
                ap_text += " " + str(obj)
            print("Attack Pattern: "+ap_text)
            # convert text to lower case
            ap_text = ap_text.lower()
            # remove numbers
            ap_text = re.sub(r'\d+', ' ', ap_text)
            # remove punctuation
            ap_text = removepunctuation(ap_text)
            # remove Whitespaces
            ap_text = removewhitespaces(ap_text)
            # convert to list
            ap_text = list(ap_text.split(" "))
            # remove empty entries
            ap_text = list(filter(None, ap_text))
            result.append(ap_text)
            data_words = list(sent_to_words(result))
            result.clear()
            # remove stop words
            data_words_nostops = remove_stopwords(data_words)
            # form bigrams and trigrams
            bigram = gensim.models.Phrases(data_words_nostops, min_count=5, threshold=100)
            trigram = gensim.models.Phrases(bigram[data_words_nostops], min_count=5, threshold=100)

            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)

            data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]
            data_words_bigrams = [x for x in data_words_bigrams if x != []]

            data_lemmatized = lemmatization(data_words_bigrams, allowed_pos=['NOUN', 'ADJ', 'VERB', 'ADV'])

            # Create Dictionary
            id2word = corpora.Dictionary(data_lemmatized)
            corpus = [id2word.doc2bow(text) for text in data_lemmatized]
            # Save corpus adn text to disk
            corpus_file_name = "CAPEC/CORPUSES/capec_corpus_" + str(capec_entry)
            corpora.MmCorpus.serialize(corpus_file_name, corpus)
            file = open("CAPEC/AP TEXTS/" + "capec_entry_" + str(capec_entry), 'wb')
            pickle.dump(data_lemmatized, file)
            print("Cleaned Text: "+str(data_lemmatized))
            print("Corpus: "+str(corpus))
            del data_lemmatized
            del corpus
            # Save dictionary to disk
            dictionary_file_name = "CAPEC/DICTIONARIES/capec_dictionary_" + str(capec_entry)
            id2word.save(dictionary_file_name)
            print("Dictionary: "+str(id2word.token2id))
            del id2word
            print("--------------------------------------------------------------------")
            capec_entry += 1

