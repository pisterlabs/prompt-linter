# Importing important libraries.
import re
import os
import nltk
# from dotenv import load_dotenv
# load_dotenv()  # UNCOMMENT THIS TO RUN LOCALLY  
import docx
import pickle
import openai
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from string import punctuation
punctuation = punctuation + "’``''"
from words_synonyms import words_synonyms
openai.api_key =  os.getenv("OPENAI_API_KEY")
stop_words = stopwords.words(stopwords.fileids())
from sklearn.feature_extraction.text import CountVectorizer
transformer = pickle.load(open("./lang-transformer.pkl", "rb"))
model = pickle.load(open("./lang-model.pkl", "rb"))


class Summarizer():
    '''
    Class containing functions to clean, format and summarize text
    Parameters:
    > self : text, type - string
    > self : churn_level float specifying percentage of original content to capture
    '''

    def __init__ (self, text:str, churn_level:float):
        self.text = text
        self.churn_level = float(churn_level)

    
    def word_sentence_tokenizer(self):
        """ 
        This function breaks text into word and sentence tokens   
        
        Parameters:

        > self : text, type -string
        > self : churn_level float specifying percentage of original content to capture

        return:

        (sent_tokens, word_tokens) : Tuple containing sentences (sentence tokens) and
        texts (text tokens) contained in text provided
        """

        sent_tokens = sent_tokenize(self.text, 'english')
        word_tokens = word_tokenize(self.text, 'english')
        return(sent_tokens, word_tokens)

        
    def summary_sorting(self, sentence_scores):
        '''
        This function selects the top n sentences based on the sentence scores
        then organizes the final sentence in asccending order of how they appeared in original text

        Parameters:

        self:churn_level - percentage of original content to capture
        sentence_scores - Dictionary containing sentences and their scores.

        return:

        final_summary : String of final / formatted summary output.
        '''

        order_sorted_sentences = []
        score_sorted_sentences = []
        sentence_score_order_tuples = []

        # multiplying churn level by number of sentences then converting to integer 
        top_n_sentences = int(self.churn_level * len(sentence_scores.keys()))

        order = 1
        # sort all sentences in descending order of their sentence_score values
        for sentence, score in sentence_scores.items():
            sentence_score_order_tuples.append((sentence, score, order))
            order += 1
        score_sorted_sentences = sorted(sentence_score_order_tuples, key=lambda tup: tup[1], reverse=True)
        # Slicing from first to top_n_sentences and appending result to produce final summary.
        top_n_slice = score_sorted_sentences[0:top_n_sentences]
        order_sorted_sentences = sorted(top_n_slice, key=lambda tup: tup[2], reverse=False)
        final_summary_list = [sentence[0] for sentence in order_sorted_sentences]
        final_sorted_summary_string = ' '.join(final_summary_list)
        return(final_sorted_summary_string)


def word_count_vec(word_tokens:list) -> tuple:
        '''
         This function produces a dictionary containing the normalized scores of each word tokens in a list
         
         Parameters:
         
         > word_tokens = [] # List of words
         
         return:

         word_frequency_scores : Tuple containing Dictionary of word tokens and their normalized scores and the most common word.

        '''
        clean_words = []
        word_frequency_scores = {}

        # Looping through to calculate word frequencies
        for word in word_tokens:
            if word.strip().lower() not in stop_words:
                if word not in punctuation:
                    clean_words.append(word)
                    if word not in word_frequency_scores:
                        word_frequency_scores[word] = 1
                    else:
                        word_frequency_scores[word] += 1
        
        # Looping through to normalize word_frequency_scores using linear / minmax scaler
        max_frequency = max(word_frequency_scores.values())
        min_frequency = min(word_frequency_scores.values())
        for word in word_frequency_scores.keys():
            word_frequency_scores[word] = (word_frequency_scores[word] - min_frequency) / (max_frequency - min_frequency)

        # Calculating top Bi-gram for topic 2
        clean_words_string = ' '.join(clean_words)
        transformer = CountVectorizer(ngram_range=(2,2))
        transformer.fit_transform([clean_words_string]) 
        topic_2 = pd.DataFrame(transformer.transform([clean_words_string]).toarray(), columns = transformer.get_feature_names_out()).T.sort_values(0, ascending=False).T.columns[0].title()
        topic_1 = max(word_frequency_scores, key=word_frequency_scores.get)
        return(word_frequency_scores, topic_1, topic_2)        


def sentence_scoring(sentence_tokens:list, word_frequency_scores:dict):
        '''
        This function calculates scores for each sentence and returns a dictionary containing sentence, score and order.
        
        Parameters:

        > sentence_tokens: List containing sentence tokens
        > word_frequency_scores: Dictionary containing word tokens and their (normalized) scores

        return:

        sentence_scores : Dictionary of sentences and their scores.

        '''
        sentence_scores = {}
        for sentence in sentence_tokens:
            for word in word_tokenize(sentence, 'english'):
                if word.lower() in word_frequency_scores.keys():
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_frequency_scores[word.lower()]
                    else:
                        sentence_scores[sentence] += word_frequency_scores[word.lower()]
        return(sentence_scores)


def string_synonym_swap(text: str) -> str:
    """
    This function converts strings to their synonyms    
    It also returns text containing CAPITAL letters as they are.

    Parameters:
    
    > text_list : Strings to be converted to synonyms

    return:

    > test_synonyms : Synonym converted string of text provided. 
    """
    synonyms = [] # final list of synonyms with first index
    text_list = text.split()
    
    for text in text_list:
        text = text.replace('”', '').replace('“', '') # This helps to conquer the problem behing reported speech "This is reported speech content being altered."
        try:
            if text.islower() and len(text) >= 3:
                synonyms.append(words_synonyms[text])
            elif text in stop_words or text in punctuation or len(text) <3:
                synonyms.append(words_synonyms[text])
            else:
                synonyms.append(text)
        except Exception:
            synonyms.append(text)
        
        # Loops through each token, checks if the token is a punctuation. if it is not a punctuation, it appends the token with a space before to the string-text body
        # if the token is a punctuation, it appens the token to the text body without a space before.
        # 'what is that?' will appear as 'what is that ? ' if this for loop didn't exist.
        string = ''
        for token in synonyms:
            if token not in punctuation:
                string += ' '+token
            else:
                string += token
    string = string.strip()
    return(string)


# Function was supposed to be for Hugging face transformer
# def abs_summary(text:str, churn_level:float) -> str:

    # Calculating minimum and maximum summary length.
    # min = int((churn_level * len(text.split())/2))
    # max = int(churn_level * len(text.split()))
    # summarizer = pipeline("summarization", model= "sshleifer/distilbart-cnn-12-6")
    # summary = summarizer(text, min_length=min, max_length=max, do_sample = False)
    # summary = """summary[0]["summary_text"].strip()"""
    # return summary


def gpt_abs_summary(text:str, churn_level:float):
    """
    Open AI extractive summary function. For this function to work, an OpenAI key needs to be supplies below
    """

    if churn_level == 0.3:
        sum_type = "short"
    elif churn_level == 0.5:
        sum_type == "medium length"
    elif churn_level == 0.7:
        sum_type = "long"
    max = int(len(text.split()))
    # This solves abstractive summary issue.
    text =" ".join(text.split())
    # Extra space in attempt to resolve EOF.
    message= "Create a "+sum_type+" summary of this for me please: "+text.replace("\n", " ").strip()+" "

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=message,
        temperature=0.7,
        max_tokens=max,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    summary = response["choices"][0]["text"].strip()
    return summary


def extract_txt(text_document) -> str:
    """
    Function to extract text from .txt file extension document

    Parameters:
    
    > Document with file extension .txt
    
    return:
    
    full_text_string : String of text contained in the .txt document provided
    """
    # Decodes byte string and cleans decoded content.
    full_text_string = text_document.read().decode("utf-8").strip()
    # Returning first 1,500 words.
    full_text_string = " ".join(full_text_string.split()[:1500])
    return(full_text_string)


def extract_docx(word_document) -> str:
    """
    Function to extract text from .docx file extension document.

    Parameters:
    
    > Document with file extension .docx
    
    return:
    
    full_text_string : String of text contained in the .docx document provided.
    """
    # Empty string variables to contain final cleaned document text.
    full_text_list = [] # sentences including empty space sentences
    full_text_string = '' # Final/clean sentences without unnecessary spaces.
    # Instantiation of word document reader object
    document = docx.Document(word_document)
    # breaking all word document objects into paragraphs
    paragraphs = document.paragraphs
    # Loop to extract text from paragraphs and append texts to list
    for paragraph in paragraphs:
        sentence = paragraph.text.strip()
        # The extra space there seperates paragraps.
        full_text_list.append(sentence+" ")
    # Loop to append sentences to sentences string. It ignors empty sentences in the list.
    for sent_ence in full_text_list:
        if sent_ence == '':
            continue
        else:
            full_text_string += sent_ence
    # Returning first 1,500 words.
    full_text_string = " ".join(full_text_string.split()[:1500])
    return(full_text_string)


def lang_detect(text:list) -> str:

    language = model.predict(transformer.transform(text))[0]
    return(language)
