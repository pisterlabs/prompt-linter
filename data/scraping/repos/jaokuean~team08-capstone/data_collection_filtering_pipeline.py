########################################## IMPORTING PACAKGES #############################

from scipy import spatial
import pandas as pd
import os
import json
import numpy
import string

import warnings
warnings.filterwarnings("ignore")


import sys  
import os
from dateutil.parser import parse


# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# PDF text extraction
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# Others
import requests
import string
import re
from pprint import pprint
from tqdm.notebook import tqdm
import io

import nltk
nltk.download('punkt')
nltk.download('stopwords')

import spacy
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=['ner'])


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

DATA_FOLDER = "data/"

########################################## DATA COLLECTION & PREPROCSSING #############################

# Text extraction from pdf
def extract_pdf(file, verbose=False):
    """
    Process raw PDF text to structured and processed PDF text to be worked on in Python.

    Parameters
    ----------
    file : textfile
        Textfile that contains raw PDF text.

    Return
    ------
    text : str
        processed PDF text if no error is throw

    """   
        
    if verbose:
        print('Processing {}'.format(file))

    try:
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        codec = 'utf-8'
        laparams = LAParams()

        converter = TextConverter(resource_manager, fake_file_handle, codec=codec, laparams=laparams)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        content = []

        for page in PDFPage.get_pages(file,
                                      pagenos, 
                                      maxpages=maxpages,
                                      password=password,
                                      caching=True,
                                      check_extractable=False):

            page_interpreter.process_page(page)

            content.append(fake_file_handle.getvalue())

            fake_file_handle.truncate(0)
            fake_file_handle.seek(0)        

        text = '##PAGE_BREAK##'.join(content)

        # close open handles
        converter.close()
        fake_file_handle.close()
        
        return text

    except Exception as e:
        print(e)

        # close open handles
        converter.close()
        fake_file_handle.close()

        return ""
    
# Text extraction from url
def extract_content(url):
    """
    Downloads PDF text content from a given URL and parse PDF to obtain processed text.

    Parameters
    ----------
    url : str
        String that contains url to desired PDF

    Return
    ------
    text : str
        processed PDF text if no error is throw

    """   
    headers={"User-Agent":"Mozilla/5.0"}

    try:
        # retrieve PDF binary stream
        r = requests.get(url, allow_redirects=True, headers=headers)
        
        # access pdf content
        text = extract_pdf(io.BytesIO(r.content))

        # return concatenated content
        return text

    except:
        return ""

    
# nlp preprocessing
def preprocess_lines(line_input):
    """
    Helper Function to preprocess and clean sentences from raw PDF text 

    Parameters
    ----------
    line_input : str
        String that contains a sentence to be cleaned

    Return
    ------
    line : str
        Cleaned sentence

    """  
    # removing header number
    line = re.sub(r'^\s?\d+(.*)$', r'\1', line_input)
    # removing trailing spaces
    line = line.strip()
    # words may be split between lines, ensure we link them back together
    line = re.sub(r'\s?-\s?', '-', line)
    # remove space prior to punctuation
    line = re.sub(r'\s?([,:;\.])', r'\1', line)
    # ESG contains a lot of figures that are not relevant to grammatical structure
    line = re.sub(r'\d{5,}', r' ', line)
    # remove emails
    line = re.sub(r'\S*@\S*\s?', '', line)
    # remove mentions of URLs
    line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', line)
    # remove multiple spaces
    line = re.sub(r'\s+', ' ', line)
    # join next line with space
    line = re.sub(r' \n', ' ', line)
    line = re.sub(r'.\n', '. ', line)
    line = re.sub(r'\x0c', ' ', line)
    
    return line
        

def remove_non_ascii(text):
    """
    Helper Function to remove non ascii characters from text
    """
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, text))

def not_header(line):
    """
    Helper Function to remove headers
    """
    return not line.isupper()

def extract_pages_sentences(nlp, text):    
    """
    Extracting text from raw PDF text and store them by pages and senteces. Raw text is also cleand by removing junk, URLs, etc.
    Consecutive lines are also grouped into paragraphs and spacy is used to parse sentences.
    Parameters
    ----------
    nlp: spacy nlp model
        NLP model to parse sentences
    text : str
        Raw PDF text

    Return
    ------
    pages_content : list of str
        A list containing text from each page of the PDF report. Page number is the index of list + 1
    
    pages_sentences : list of list
        A list containing lists. Page number is the index of outer list + 1. Inner list contains sentences from each page
 
    """  
    MIN_WORDS_PER_PAGE = 500
    
    pages = text.split('##PAGE_BREAK##')
    #print('Number of Pages: {}'.format(len(pages)))

    lines = []
    for i in range(len(pages)):
        page_number = i + 1
        page = pages[i]
        
        # remove non ASCII characters
        text = remove_non_ascii(page)
        
        # if len(text.split(' ')) < MIN_WORDS_PER_PAGE:
        #     print(f'Skipped Page: {page_number}')
        #     continue
        
        prev = ""
        for line in text.split('\n\n'):
            # aggregate consecutive lines where text may be broken down
            # only if next line starts with a space or previous does not end with dot.
            if(line.startswith(' ') or not prev.endswith('.')):
                prev = prev + ' ' + line
            else:
                # new paragraph
                lines.append(prev)
                prev = line

        # don't forget left-over paragraph
        lines.append(prev)
        lines.append('##SAME_PAGE##')
        
    lines = '  '.join(lines).split('##SAME_PAGE##')
    
    # clean paragraphs from extra space, unwanted characters, urls, etc.
    # best effort clean up, consider a more versatile cleaner
    
    pages_content = []
    pages_sentences = []

    for line in lines[:-1]: # looping through each page
        
        line = preprocess_lines(line)       
        pages_content.append(str(line).strip())

        sentences = []
        # split paragraphs into well defined sentences using spacy
        for part in list(nlp(line).sents):
            sentences.append(str(part).strip())

        #sentences += nltk.sent_tokenize(line)
            
        # Only interested in full sentences and sentences with 10 to 100 words. --> filter out first page/content page
        sentences = [s for s in sentences if re.match('^[A-Z][^?!.]*[?.!]$', s) is not None]
        sentences = [s.replace('\n', ' ') for s in sentences]
        
        pages_sentences.append(sentences)
        
    return pages_content, pages_sentences #list, list of list where page is index of outer list


def preprocessing(report):
    """
    Lemmatize,lowercase and remove stopwords for pages of a report
    
    Parameters
    ----------
    report: list of str
        A list containing text from each page of the PDF report. Page number is the index of list + 1

    Return
    ------
    report_pages : list of str
        A list containing processed text from each page of the PDF report. Page number is the index of list + 1
    
    """  
    
    report_pages = []

    def para_to_sent(para):
        """
        Helper function to split paragraphs into well defined sentences using spacy
        """
        sentences = []
        for part in list(nlp(para).sents):
            sentences.append(str(part).strip())
        return sentences

    def remove_stopwords(texts):
        """
        Helper function to remove stopwords from sentence
        """
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        """
        Helper function to lemmatize text in sentence
        """
        texts_out = []
        doc = nlp(texts) 
        texts_out.append(" ".join([token.lemma_ for token in doc]))
        return texts_out
    
    for page in report:

        sentences = para_to_sent(page.lower())

        # Do lemmatization keeping only noun, adj, vb, adv
        page_data = []
        for sentence in sentences : 
            data_lemmatized = lemmatization(sentence, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
            page_data.extend(data_lemmatized)
        page_para_lemma = "".join(page_data)
        
        report_pages.append(page_para_lemma)
    
    return report_pages



def lemmatization(text_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # lemmatize text in sentence
    """https://spacy.io/api/annotation"""
    texts_out = []
    for texts in text_list:
        texts = texts.lower()
        texts_out.append(" ".join([token.lemma_ for token in nlp(texts)]))
    return texts_out


def filter_report_highLevel(report):
    """
    Page filter to filter report for only relevant pages with decarbonisation related words.
    Two types of word filters: direct and indirect. Direct contains words that are directly related to decarbonisation while indirect contains other relevant decarbonisation information.
    
    Parameters
    ----------
    report: list of str
        A list containing text from each page of the PDF report. Page number is the index of list + 1

    Return
    ------
    filtered_report_direct : dict of {int : str}
        A dictionary that contains relevant pages obtained using direct filter. The key is the page number and value is the text on the page. 
    
    filtered_report_indirect : dict of {int : str}
        A dictionary that contains relevant pages obtained using indirect filter. The key is the page number and value is the text on the page.     
    """  
    
    # list of words used to filter
    relevant_terms_directFilter = set(["carbon","co2","environment","GHG emissions","Greenhouse Gas","carbon footprint","carbon emissions","Scope 1","Scope 2",
                               "Scope 3", "WACI","Carbon Intensity","carbon pricing","net-zero","metrics and targets","TCFD",
                                "sustainability goals","decarbonisation","climate",'energy', 'emission', 'emissions', 'renewable', 'carbon', 'fuel', 'power', 
                               'green', 'gas', 'green energy', 'sustainable', 'climate', 'sustainability', 'environmental', 'environment', 'GHG', 
                               'decarbon', 'energy consumption', 'paper consumption','water consumption', 'carbon intensity', 'waste management', 'electricity consumption', 
                                'cdp', 'global warming', 'business travel','climate solutions', 'decarbonization', 'cvar', 'climate value-at-risk','waste output'])
    relevant_terms_combinationA = ["emissions","exposure","carbon related","esg","sustainable","green","climate sensitive","impact investing", "investment framework", 'msci', 'ftse', 'responsible investing', 'responsible investment','transition']
    relevant_terms_combinationB = ["portfolio","assets","AUM","investment","financing","ratings","revenue","bond","goal","insurance", "equity", "swap", "option", "portfolio holdings", "risk management",'financial products']
    relevant_terms_combinationC = ["net zero","carbon footprint","CO2","carbon","oil","coal", "gas", "fossil fuel","green"]
    relevant_terms_combination_directFilter_lem = lemmatization(relevant_terms_directFilter)
    relevant_terms_combinationA_lem = lemmatization(relevant_terms_combinationA)
    relevant_terms_combinationB_lem = lemmatization(relevant_terms_combinationB)
    relevant_terms_combinationC_lem = lemmatization(relevant_terms_combinationC)
    
    
    filtered_report_direct = {}
    filtered_report_indirect = {}
    for i in range(len(report)):
        page = report[i]
        page_number = i + 1
        no_words = len(page.split(" "))
        
        # filter for pages that contain at least 3 words from the relevant_terms_combination_directFilter_lem list
        if sum(map(page.__contains__, relevant_terms_combination_directFilter_lem)) > 2:
            filtered_report_direct[page_number] = page
        
        # filter for pages that contain at least 1 word (relevant_terms_combinationC_lem AND relevant_terms_combinationA_lem) OR (relevant_terms_combinationC_lem AND  relevant_terms_combinationB_lem)
        elif (any(map(page.__contains__, relevant_terms_combinationA_lem)) and any(map(page.__contains__, relevant_terms_combinationC_lem))) or (any(map(page.__contains__, relevant_terms_combinationB_lem)) and any(map(page.__contains__, relevant_terms_combinationC_lem))):
            filtered_report_indirect[page_number] = page
    
    return filtered_report_direct,filtered_report_indirect


def is_number(string): 
    """
    Helper function that checks if a string contains numbers
    """
    test_str = string
    # next() checking for each element, reaches end, if no element found as digit
    res = True if next((chr for chr in test_str if chr.isdigit()), None) else False
    return res

def is_date(string):
    """
    Helper function that checks if a string contains dates such as "september"/"Monday at 12:01am"/1999 etc
    """
    if re.match('.*([1-2][0-9]{3})', string) != None:
        return True
    return False

def filter_report_numbers(filtered_report): 
    """
    Page filter to filter the filtered report for pages with numbers that are not dates as to track decarbonisation progress, we need numbers.
    
    Parameters
    ----------
    filtered_report: dict of {int : str}
        A dictionary that contains relevant pages obtained using page filters. The key is the page number and value is the text on the page.
        
    Return
    ------
    filtered_report_numbers : dict of {int : str}
        A dictionary that contains relevant pages with numbers. The key is the page number and value is the text on the page. 
    
    """ 
        
    filtered_report_numbers = {}
    for page_number,page in filtered_report.items():
        # remove all dates from page first
        page_no_date = " ".join([word for word in page.split(" ") if is_date(word) == False])
        # retain pages with numbers
        if is_number(page_no_date):
            filtered_report_numbers[page_number] = page           
    return filtered_report_numbers



def filter_tables(filtered_report):    
    """
    Page filter to filter the filtered report for pages with tables to aid table detection in downstream tasks. This is done by checking if page contains at least 10 numbers + 1 units
    
    Parameters
    ----------
    filtered_report: dict of {int : str}
        A dictionary that contains relevant pages with numbers. The key is the page number and value is the text on the page.
        
    Return
    ------
    filtered_report_numbers : dict of {int : str}
        A dictionary that contains relevant pages with tables. The key is the page number and value is the text on the page. 
    
    """ 
    
    
    units = ['tonnes', 'tons', 'kwh', 'kg', 'kilogram', 'kilowatt hour', 'gigajoules', 'gj', 'litre', 'liter', 'co2e', 'tco2e', 'tco2', 'mwh', 'megawatt hour', 'gwh', 'gigawatt hour', '%', 'cubic metres', 'cm3', 'm3', 'per employee','co2']
    filtered_report_numbers = {}
    for page_number,page in filtered_report.items():
        no_numbers = 0
        units_flag = 0
        for word in page.split(" "):
            try: 
                float(word)
                if is_date(word) == False:
                    no_numbers += 1
            except: 
                if any(char.isdigit() for char in word):
                    no_numbers += 1
        for unit in units:
            if unit in page:
                units_flag += 1     
        if (no_numbers >= 10) & (units_flag >= 1):
            filtered_report_numbers[page_number] = page
    return filtered_report_numbers



# new pdf is saved in a new json file & also appended to existing file
# report_url can be either from internet : url downloaded=False OR local : path to pdf downloaded=True

def upload_pdf(report_url,report_company,report_year,downloaded=False):
    """
    Main Function to run to obtain PDF and parse,clean and filter the raw PDF text.
    
    Parameters
    ----------
    report_url: str
        If downloaded=False, report is from the internet and report_url is a URL string to the PDF.
        If downloaded=True, report is from local machine and report_url is the file path to the PDF.
    
    report_company: str
        Company name
    
    report_year: str
        Year of report
        
    downloaded : bool
        Whether report needs to be downloaded from Internet or not
        
    Return
    ------
    file_path : str
        File path of the output file for subsequent downstream tasks if PDF could be obtained. Else return empty string
    
    """ 
    
    # check if report is from Internet or Local Machine
    if downloaded == True:
        with open(report_url,"rb") as inputfile:
            report_content = extract_pdf(inputfile)
    else:
        report_content = extract_content(report_url)
        
    report = {'company':report_company, 'year':report_year,'url':report_url, 'content':report_content}
    
    # check if report content is empty
    if report["content"] == "":
        print("Unable to get PDF")
        return ""
    
    # if report content is not empty process and filter report
    else:
        print("PARSING PDF TO TEXT")
        report_pages, report_sentences = extract_pages_sentences(nlp, report['content'])
        report["report_pages"] = report_pages
        report["report_sentences"] = report_sentences
        report_pages_preprocessed = preprocessing(report["report_pages"])
        report_sentences_preprocessed = [preprocessing(page) for page in report["report_sentences"]]
        report["report_pages_preprocessed"] = report_pages_preprocessed 
        report["report_sentences_preprocessed"] = report_sentences_preprocessed 
        
        # filter pages by words and numbers
        filtered_report_direct_highLevel,filtered_report_indirect_highLevel = filter_report_highLevel(report["report_pages_preprocessed"])
        filtered_report_pages_direct_numbers = filter_report_numbers(filtered_report_direct_highLevel)
        filtered_report_pages_indirect_numbers = filter_report_numbers(filtered_report_indirect_highLevel)
        report["filtered_report_pages_direct"] = filtered_report_pages_direct_numbers
        report["filtered_report_pages_indirect"] = filtered_report_pages_indirect_numbers
        index_direct = [page_no-1 for page_no in filtered_report_pages_direct_numbers.keys()] 
        index_indirect = [page_no-1 for page_no in filtered_report_pages_indirect_numbers.keys()] 

        # filter sentences by words and numbers
        filtered_report_sentences_direct_numbers = {}
        for page in filtered_report_pages_direct_numbers.keys():
            filtered_report_sentences_direct_numbers[page] = report["report_sentences_preprocessed"][page-1]
        filtered_report_sentences_indirect_numbers = {}
        for page in filtered_report_pages_indirect_numbers.keys():
            filtered_report_sentences_indirect_numbers[page] = report["report_sentences_preprocessed"][page-1]
        report["filtered_report_sentences_direct"] = filtered_report_sentences_direct_numbers
        report["filtered_report_sentences_indirect"] = filtered_report_sentences_indirect_numbers

        # filter pages for tables
        report["filtered_report_tables_direct"] = filter_tables(filtered_report_pages_direct_numbers)
        report["filtered_report_tables_indirect"] = filter_tables(filtered_report_pages_indirect_numbers)
        
        file_path = DATA_FOLDER + "new_report/" + report_company + report_year+'.json'
        
        with open(file_path, "w") as outfile:  
            json.dump(report, outfile)
        
        print("DONE PARSING PDF TO TEXT")
        
        return file_path







####################################### BERT AS A SERVICE FILTERING #################################
# instantiate BERT as a Service
print("INSTANTIATING BERT AS A SERVICE")
from bert_serving.client import BertClient
bc = BertClient(check_length=False)


# helper functions
def remove_punc(s):
    """
    Helper function to remove all punctuations except "$&%" from sentence
    """
    exclude = string.punctuation
    final_punc = ''.join(list(i for i in exclude if i not in ['%', '$', '&']))
    s = ''.join(ch for ch in s if ch not in list(final_punc))
    return s

def cosine_distance(s1,s2):
    """
    Helper function to calculate cosine similarity of 2 sentence embeddings
    """
    return 1 - spatial.distance.cosine(s1, s2)

# indentation
def create_bert_embeddings(jsonfile):
    """
    Create word embeddings for highly relevant sentences
    
    Parameters
    ----------
    jsonfile: dict of {str : list of str}
        A dictionary that contains a company's report details, preprocessed text and relevant pages with highly relevant sentences. 
        
    Return
    ------
    jsonfile: dict of {str : list of str}
        A dictionary that contains a company's report details, preprocessed text, relevant pages with highly relevant sentences and the sentence embeddings of the highly relevant sentences.  
    
    """ 
    
    print("CREATE BERT EMBEDDINGS FOR RELEVANT SENTENCES")
    if jsonfile["bert_relevant_sentences_direct_original"] != {}:
        embeddings_dict = {}
        for page,sentences in jsonfile["bert_relevant_sentences_direct_original"].items():
            embeddings_dict[page] = []
            for sentence in sentences:
                sentence_encoding = list(bc.encode([sentence])[0])
                embeddings_dict[page].append(list(map(lambda x: numpy.float64(x),sentence_encoding)))
            jsonfile["bert_relevant_sentences_direct_original_embeddings"] = embeddings_dict
    else:
        jsonfile["bert_relevant_sentences_direct_original_embeddings"] = {}

    if jsonfile["bert_relevant_sentences_indirect_original"] != {}:
        embeddings_dict = {}
        for page,sentences in jsonfile["bert_relevant_sentences_indirect_original"].items():
            embeddings_dict[page] = []
            for sentence in sentences:
                sentence_encoding = list(bc.encode([sentence])[0])       
                embeddings_dict[page].append(list(map(lambda x: numpy.float64(x),sentence_encoding)))
            jsonfile["bert_relevant_sentences_indirect_original_embeddings"] = embeddings_dict
    else:
        jsonfile["bert_relevant_sentences_indirect_original_embeddings"] = {}
        
    return jsonfile


# to process 1 json
def bert_filtering(file_path):
    """
    Main function to filter for highly relevant sentences by comparing cosine similarity of all sentences from filtered report with highly relevant sentences determined by us.
    
    Parameters
    ----------
    file_path: str
        File path to the json that contains the filtered PDF report.
        
    Return
    ------
    output_path: str
        File path to the output json that contains highly relevant sentences and their bert sentence embeddings. 
    
    """     
    
    
    with open(file_path,) as inputfile:
        json_file = json.load(inputfile)
        json_file = [json_file]
    fi_list = []
    
    # step 1: obtain relevant fields from input json required for bert and text similarity
    for fi in json_file:
        fi_dict = {}
        fi_direct_dict = {}
        for page in fi["filtered_report_pages_direct"].keys():
            page = int(page)
            fi_direct_dict[page] = list(map(lambda x : remove_punc(x) ,fi["report_sentences"][page-1]))
        fi_indirect_dict = {}
        for page in fi["filtered_report_pages_indirect"].keys():
            page = int(page)
            fi_indirect_dict[page] =  list(map(lambda x : remove_punc(x) ,fi["report_sentences"][page-1]))      
        fi_dict["company"] = fi["company"] #identifier
        fi_dict["year"] = fi["year"] #identifier
        fi_dict["url"] = fi["url"]
        fi_dict["filtered_report_pages_direct_bert"]  = fi_direct_dict
        fi_dict["filtered_report_pages_indirect_bert"]  = fi_indirect_dict
        fi_list.append(fi_dict)
        
    
    # list of highly relevant sentences determined by us
    relevant_sentences = ['In 2019, Citi financed $74 million of subordinate lien bonds that were certified green, given the projects environmental aspects.', 
                'In addition, our cogeneration plant, fueled by natural gas, will produce heat and electricity on-site, reducing the building\'s carbon footprint by 34 percent.',
                'These efforts reduced energy consumption by more than 2,100 metric tons (mt) of carbon dioxide equivalents (CO2e) during the one-year challenge.',
                'The companies in our equity portfolio emitted around 133 tonnes of CO2 -equivalents for every million US dollars of revenue.', 
                'The equity portfolio’s carbon intensity was 9 percent below that of the benchmark index.',
                'A total of 106 companies that produce certain types of weapon, tobacco or coal, or use coal for power production, are currently excluded from the fund', 
                'For public and private assets, excluding cash and non-equity derivatives as they were not reported in 2019, our year-overyear portfolio weighted average carbon intensity was reduced by approximately 23%.',
                'Having met these targets, we have set new, more ambitious ones: to reduce the Fund’s emissions intensity by 40% and fossil fuel reserves by 80% by 2025.',
                'The carbon footprint of the non-listed companies was 0.6 tCO₂e per million SEK invested',
                'Energy consumption and carbon emissions per unit area were 149 kWh/ m² and 0.037 tCO₂e/m², which means a reduction of 9 per cent and 12 per cent, respectively.',
                'The carbon intensity (CO2 equivalent tons per million yen of sales) of GPIF’s equity and corporate bond portfolio decreased by 15.3%, from 2.29 tons to 1.94 tons, in the space of a year.'
                'Based on our percentage holdings in each company, the total emissions of the equity portfolio were 108 million tonnes of CO2 - equivalents in 2019.',
                'The carbon footprint of the companies in our equity portfolio',
                'The companies in our equity portfolio emitted around 156 tonnes of CO2 -equivalents for every million US dollars (USD) of revenue.'
                'The carbon intensity of the companies in the equity portfolio and the benchmark index decreased by 16 and 17 percent respectively from 2018 to 2019.',
                'We are focused on supporting the goal of net zero greenhouse gas emissions by 2050, in line with global efforts to limit warming to 1.5°C. ',
                'quantitative target for ESG-themed investments and finance of ¥700 billion ',
                'Commit to reduce investment carbon footprint by',
                'esg investing', 'green bonds', 'Green Investment target', 'Achieve 100% renewable electricity by 2025']
    
    # encode these sentences using BERT
    relevant_sentences_embeddings = bc.encode(relevant_sentences)

    # step 2: encode all sentences and compare cosine similarity. only keep sentences with a cosine similarity higher than 0.7357.
    json_list = fi_list
    for fi_index in range(len(json_list)): #remove
        fi = json_list[fi_index]
        page_relevant_sentences = {}
        page_relevant_sentences_original = {}
        for page_number, page in fi["filtered_report_pages_direct_bert"].items():
            relevant_sentences = []
            relevant_sentences_original = []
            for sentence_index in range(len(page)):
                original_sentence = json_file[fi_index]["report_sentences"][page_number-1][sentence_index]
                sentence = page[sentence_index]
                sentence_encoding = bc.encode([sentence])[0]
                for relevant_sentence in relevant_sentences_embeddings:
                    cosine_similarity = cosine_distance(sentence_encoding,relevant_sentence)
                    if cosine_similarity >= 0.7357 : # tentative
                        relevant_sentences.append(sentence)
                        relevant_sentences_original.append(original_sentence)
                        break
            if len(relevant_sentences) != 0 :
                page_relevant_sentences[page_number] = relevant_sentences
                page_relevant_sentences_original[page_number] = relevant_sentences_original

        json_list[fi_index]["bert_relevant_sentences_direct_original"] = page_relevant_sentences_original # sentences used for model train


        page_relevant_sentences_indirect = {}
        page_relevant_sentences_indirect_original = {}
        for page_number, page in fi["filtered_report_pages_indirect_bert"].items():
            relevant_sentences = []
            relevant_sentences_original = []
            for sentence_index in range(len(page)):
                original_sentence = json_file[fi_index]["report_sentences"][page_number-1][sentence_index]
                sentence = page[sentence_index]
                sentence_encoding = bc.encode([sentence])[0]
                for relevant_sentence in relevant_sentences_embeddings:
                    cosine_similarity = cosine_distance(sentence_encoding,relevant_sentence)
                    if cosine_similarity >= 0.7357 : # tentative maybe must b smilar to most terms?
                        relevant_sentences.append(sentence)
                        relevant_sentences_original.append(original_sentence)
                        break
            if len(relevant_sentences) != 0 :
                page_relevant_sentences_indirect[page_number] = relevant_sentences
                page_relevant_sentences_indirect_original[page_number] = relevant_sentences_original

        json_list[fi_index]["bert_relevant_sentences_indirect_original"] = page_relevant_sentences_indirect_original # sentences used to train
        

    #step 3 create embeddings for highly relevant sentences
    final_json_embeddings = create_bert_embeddings(json_list[0])
    output_path = file_path[:-5] + "_BERT_embeddings.json"
    

    with open(output_path, "w") as outfile:  
        json.dump(final_json_embeddings, outfile)

    return output_path




# ##################### PIPELINE #######################################################

# # for new url
# def new_url_run(report_url,report_company,report_year,downloaded=False):
#     # new json generated in "../data/sustainability_reports_new"
#     report_output_file_path = upload_pdf(report_url,report_company,report_year,downloaded)
#     # new BERT_embeddings_json generated in "../data/sustainability_reports_new"
#     report_bert_output_file_path = bert_filtering(report_output_file_path)
       
    
    

