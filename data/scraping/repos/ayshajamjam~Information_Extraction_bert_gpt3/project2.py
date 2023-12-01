import string
import sys
import re
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en import English
import itertools
from itertools import permutations
import os
import openai
import json
from operator import itemgetter
from prompts import *
import random

iteration_num = 0

developer_key, engine_key, openai_key, method, query = "", "", "", "", ""
relation = 0
threshold = 0.0
k = 10

relation_in_words = ""

x = set() # the set of extracted tuples
y = {}
queried_relations = set()   # set of relations used to query

def extract_arguments(args):
    # query URL = python3 project2.py [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <r> <t> <q> <k>
    if (len(args) <= 9):
        print("Invalid number of arguments provided: ", len(args))
        print("Arguments should contain: project2.py [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <r> <t> <q> <k>")
        return
    
    # Extract arguments
    num_arguments = len(args)

    pattern = re.compile(r'[^A-Za-z0-9 ]+')
    method = re.sub(pattern, '', args[1])
    developer_key = args[2]
    engine_key = args[3]
    openai_key = args[4]
    relation = args[5]
    threshold = args[6]
    query = ""
    for i in range(7, len(args) - 1):
        query += args[i] + " "
    query = re.sub(pattern, '', query).lower()
    k = args[len(args) - 1]

    global relation_in_words
    relation_in_words = ""

    if(relation == '1'):
        relation_in_words = "Schools_Attended"
    elif(relation == '2'):
        relation_in_words = "Work_For"
    elif(relation == '3'):
        relation_in_words = "Live_In"
    elif(relation == '4'):
        relation_in_words = "Top_Member_Employees"
    
    # Print arguments
    print("\nParameters:")
    print("Client key=\t{}\nEngine key=\t{}\nOpenAI key=\t{}".format(developer_key, engine_key, openai_key))
    print("Method=\t{}\nRelation=\t{}\nThreshold=\t{}\nQuery=\t{}\n# of Tuples=\t{}\n".format(method, relation_in_words, threshold, query, k))
    print("Loading necessary libraries; This should take a minute or so ...)")

    return developer_key, engine_key, openai_key, method, int(relation), float(threshold), query, int(k)

def get_top_ten_webpages(query, developer_key, engine_key):

    service = build(
        "customsearch", "v1", developerKey=developer_key
    )

    res = (
        service.cse()
        .list(
            q=query,
            cx=engine_key,
        )
        .execute()
    )

    number_of_search_results = 10

    urls = []

    # Create list of top 10 urls
    for i in range(number_of_search_results):
        # Res (dict) -> res['items'] (list) -> res['items'][0](dict)
        try:
            url = res["items"][i]["link"]
            urls.append(url)
        except:
            print(res)
            print("Poorly formed query")
    
    return urls

def process_url(url):
    print("Fetching text from url ...")

    # Send HTTP request to retrieve HTML content of webpage
    try:
        r = requests.get(url, timeout=7, verify=True)
        r.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error")
        print(errh.args[0])
        return 0
    except requests.exceptions.ReadTimeout as errrt:
        print("Time out")
        return 0
    except requests.exceptions.ConnectionError as conerr:
        print("Connection error")
        return 0
    except requests.exceptions.RequestException as errex:
        print("Exception request")
        return 0

    # page = r.content # raw HTML content of webpage
    print('successful')
    # Parse the HTML content using html parser (html5lib)
    soup = BeautifulSoup(r.content, 'html5lib')

    # Get rid of tags with unneccessary information
    for data in soup(['style', 'script', 'header', 'footer', 'nav', 'meta']):
        data.decompose()

    content = soup.get_text()

    # Truncate plain text to 10,000 chars if neccessary
    if(len(content) > 10000):
        print("Trimming webpage content from {} to 10000 characters".format(len(content)))
        content = content[0:10000]

    print("Webpage length (num characters): ", len(content))
    # print(content)

    return content

def text_to_sentences(text):
    nlp = English()
    nlp.add_pipe("sentencizer")

    doc = nlp(text)
    sentences_string = " "
    sentences_list = []
    for i, sent in enumerate(doc.sents):
        sent = str(sent).strip()
        # sent = str(sent).translate(str.maketrans('', '', string.punctuation))
        sent = re.sub(u'\xa0', ' ', sent)
        sent = sent.replace('\t+', ' ')
        sent = sent.replace('\n+', ' ')
        sent = re.sub('  +', ' ', sent)
        sent = sent.replace('\u200b', ' ') 

        sent = " ".join(sent.split())
        sentences_string += sent + " "
        sentences_list.append(sent)

    return sentences_string, sentences_list

def annotate(sentence, relation):

    named_entities_labels = set()
    entities_of_interest = []

    # Potential relations
    if(relation == 1):
        entities_of_interest = ["PERSON", "ORGANIZATION"] 
    elif(relation == 2):
        entities_of_interest = ["PERSON", "ORGANIZATION"] 
    elif(relation == 3):
        entities_of_interest = ["PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    elif(relation == 4):
        entities_of_interest = ["ORGANIZATION", "PERSON"] 

    # print("Relation: ", relation)
    # print(entities_of_interest)

    # Load spacy model
    nlp = spacy.load("en_core_web_sm")

    # Apply spacy model to raw text (to split to sentences, tokenize, extract entities etc.)
    doc = nlp(str(sentence))

    # Find named_entities in sentence that correspond to user-defined relation selection
    from spacy_help_functions import get_entities, create_entity_pairs
    named_entities = get_entities(doc, entities_of_interest)
    # print(named_entities, '\n')

    if(len(list(doc.sents)) !=  1):
        return 0

    # print(list(doc.sents))

    # Construct candidate entity pairs
    candidate_pairs = create_entity_pairs(list(doc.sents)[0], entities_of_interest)
    
    # Find named_entities in sentence that correspond to user-defined relation selection
    filtered_pairs = []
    for pair in candidate_pairs:
        if((relation == 1 or relation == 2) and pair[1][1] == 'PERSON' and pair[2][1] == 'ORGANIZATION'):
            filtered_pairs.append(pair)
        elif(relation == 3 and pair[1][1] == 'PERSON' and (pair[2][1] == 'LOCATION')):
            filtered_pairs.append(pair)
        elif(relation == 4 and pair[1][1] == 'ORGANIZATION' and pair[2][1] == 'PERSON'):
            filtered_pairs.append(pair)
    
    # Filter entity pairs by relation
    if(len(filtered_pairs) == 0):
        return 0

    return 1   

# Feed into extractSpanbert all filtered pairs belonging to one sentence
def extractSpanbert(sentences, spanbert, relation, threshold):

    entities_of_interest = []
    internal_relation = ""

    # Potential relations
    if(relation == 1):
        entities_of_interest = ["PERSON", "ORGANIZATION"] 
        internal_relation = "per:schools_attended"
    elif(relation == 2):
        entities_of_interest = ["PERSON", "ORGANIZATION"] 
        internal_relation = "per:employee_of"
    elif(relation == 3):
        entities_of_interest = ["PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
        internal_relation = "per:cities_of_residence"
    elif(relation == 4):
        entities_of_interest = ["ORGANIZATION", "PERSON"] 
        internal_relation = "org:top_members/employees"

    # Load spacy model
    nlp = spacy.load("en_core_web_sm")

    # Apply spacy model to raw text (to split to sentences, tokenize, extract entities etc.)
    doc = nlp(sentences)

    # Extract relations
    from spacy_help_functions import extract_relations
    relations, total_extracted, num_relevant_sents = extract_relations(doc, spanbert, internal_relation, entities_of_interest, threshold)
    return dict(relations), total_extracted, num_relevant_sents

def get_openai_completion(prompt, model, max_tokens, temperature = 0.2, top_p = 1, frequency_penalty = 0, presence_penalty =0):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        timeout=5
    )
    response_text = response['choices'][0]['text']
    return response_text

def main(new_query=None):

    global developer_key, engine_key, openai_key, method, query
    global relation, threshold, k
    global seen_urls
    global x
    global y
    global iteration_num

    if(new_query == None):
        # Initialization: extract from arguments
        developer_key, engine_key, openai_key, method, relation, threshold, query, k = extract_arguments(sys.argv)
    else:
        query = new_query

    queried_relations.add(query.strip().lower())

    iteration_num += 1
    print("\n=========== Iteration: {} - Query: {} ===========".format(iteration_num, query))

    # Top-10 webpages for seed query
    top_ten_webpages = get_top_ten_webpages(query, developer_key, engine_key)

    # openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_key
    models_li= openai.Model.list()

    model = 'text-davinci-003'
    max_tokens = 100
    temperature = 0.2
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0

    # Load pre-trained SpanBERT model
    from spanbert import SpanBERT 
    spanbert = SpanBERT("./pretrained_spanbert")

    seen_urls = set()

    for i, url in enumerate(top_ten_webpages):
        
        new_extracted = 0
        num_relevant_sents = 0
        total_extracted = 0

        print('URL ({}/10): {} '.format(i+1, url))

        # skip seen urls
        if url in seen_urls:
            print("URL already visited")
            continue
        seen_urls.add(url)

        # Extract information from URL
        text = process_url(url)
        if(text == 0):
            continue

        # spaCy: Split text into sentences 
        sentences_string, sentences_list = text_to_sentences(text)
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentences_string)
        num_sentences = len([s for s in doc.sents])

        print("Annotating the webpage using spacy...")
        print("Extracted {} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...".format(num_sentences))

        # Relation Extraction
        if (method == 'spanbert'):
            # spaCy-Spanbert: Extract named entities
            # Filter sentences
            relations, total_extracted, num_relevant_sents = extractSpanbert(sentences_string, spanbert, relation, threshold)
            for j, rel in enumerate(relations.keys()):
                y[rel] = relations[rel]
                new_extracted = new_extracted + 1
                print('\n')
  
        elif (method == 'gpt3'):
            # Filter sentences
            num_sentences = len(sentences_list)
            for i, sentence in enumerate(sentences_list):
                if((i+1) % 5 == 0):
                    print("        Proccessed {}/{} sentences".format(i+1, len(sentences_list)))
                rel_sentence = annotate(sentence, relation)
                if(rel_sentence != 0):
                    num_relevant_sents += 1

                    prompt_text = ""
                    if(relation == 1):
                        prompt_text = prompt1.format(sentence)
                    elif(relation == 2):
                        prompt_text = prompt2.format(sentence)
                    elif(relation == 3):
                        prompt_text = prompt3.format(sentence)
                    elif(relation == 4):
                        prompt_text = prompt4.format(sentence)
                    response_text = get_openai_completion(prompt_text, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)
                    
                    # Clean up response
                    response_text = response_text.strip().split('\n')
                    # print(i, ")")
                    # print(response_text)

                    for tup in response_text:
                        tup = tup.replace('(', '')
                        tup = tup.replace(')', '')
                        tup = tup.replace("'", '')
                        res = tuple(map(str, tup.split(', ')))
                        if(len(res) != 2):
                            continue
                        left = re.sub('[^0-9a-zA-Z\s\-&]+', '', res[0])
                        right = re.sub('[^0-9a-zA-Z\s\-&]+', '', res[1])
                        res = tuple([left, right])
                        # print(res)
                        total_extracted += 1
                        print("                === Extracted Relation ===")
                        print("                Sentence: ", sentence)
                        print("                Subject: {} ; Object: {}".format(res[0], res[1]))
                        if res in x:
                            print("                Ignoring duplicate")
                        else:
                            x.add(res)
                            new_extracted += 1
                            print("                Adding to set of extracted relations")
                        print("                ==========\n")
                    print('\n')

        # Count number of extracted relations from this URL
        print("        Extracted annotations {} out of total {} sentences".format(num_relevant_sents, num_sentences))
        print("        Relations extracted from this website: {} (Overall: {})".format(new_extracted, total_extracted))

    if(method == 'gpt3'):
        print("\n================== ALL RELATIONS for {} ( {} ) =================".format(relation_in_words, len(x)))
        x_sorted = sorted(x, key=itemgetter(0))   # sort extracted relations by subject
        for rel in x_sorted:
            print("Subject: {}             | Object: {}".format(rel[0], rel[1]))
    
        # Check if we need to do another iteration
        if len(x) >= k:
            print("Total # of iterations = {}\n".format(iteration_num))
        else:
            print("\nNeed to do another iteration")
            new_query = ""
            while True:
                rel_num = random.randint(0, len(x_sorted) - 1)
                new_query = (x_sorted[rel_num][0]).strip() + ' ' + (x_sorted[rel_num][1]).strip()
                if(len(x_sorted) != 1 and new_query.lower() in queried_relations):
                    continue
                else:
                    break
            if (new_query == "" or new_query.lower() in queried_relations):
                print("No further queries to explore")
            else:
                print(query)
                print(new_query)
                print(query == new_query)
                main(new_query)
    else:
        y_sorted = sorted(y.items(), key=lambda x:x[1], reverse=True)
        if len(y) >= k:
            print("\n================== ALL RELATIONS for {} ( {} ) =================".format(relation_in_words, k))
            j = 0
            for rel in y_sorted:
                if j == k:
                    break
                print("Confidence: {}             | Subject: {}             | Object: {}".format(rel[1], rel[0][0], rel[0][2]))
                j = j + 1
        else:
            print("\n================== ALL RELATIONS for {} ( {} ) =================".format(relation_in_words, len(y)))
            for rel in y_sorted:
                print("Confidence: {}             | Subject: {}             | Object: {}".format(rel[1], rel[0][0], rel[0][2]))
        
        # Check if we need to do another iteration
        if len(y) >= k:
            print("Total # of iterations = {}\n".format(iteration_num))
        else:
            print("\nNeed to do another iteration")
            new_query = ""
            while True:
                rel_num = random.randint(0, len(y) - 1)
                new_query = (y_sorted[rel_num][0][0]).strip() + ' ' + (y_sorted[rel_num][0][2]).strip()
                if(len(y) != 1 and new_query.lower() in queried_relations):
                    continue
                else:
                    break
            if (new_query == "" or new_query.lower() in queried_relations):
                print("No further queries to explore")
            else:
                main(new_query)

if __name__ == "__main__":
    main()