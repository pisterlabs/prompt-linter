#!/usr/bin/python
# -*- coding: utf-8 -*-

import concurrent.futures
import os
import re
import urllib
from concurrent.futures.thread import ThreadPoolExecutor
from os.path import exists
from time import sleep, time

import nlpcloud  # NLP Cloud Playground https://www.nlpcloud.com
import nltk.data  # NLP sentence parser used to remove any duplicate word for word sentence output from AI response
import openai  # OpenAI https://www.openai.com
import pandas as pd
import requests 
# https://pypi.org/project/lxml/
# https://lxml.de/lxmlhtml.html
# pip install beautifulsoup4
# https://lxml.de/elementsoup.html
from lxml.html.soupparser import fromstring 
from lxml.html.clean import Cleaner
# https://pypi.org/project/requests-html/
from requests_html import HTML, HTMLSession 

# parallelism makes concurrent simulataneous calls to the AI Engine
# From those responses, the best text is determined by length 
# If you want to choose the best of 3 calls for the same next scene, then enter 3 below.

class ZestorHelper:

    open_ai_api_key = os.getenv('OPENAI_API_KEY') # not needed, but for clarity
    nlp_cloud_api_key = os.getenv('NLPCLOUD_API_KEY') 

    AI_ENGINE_NLPCLOUD = 'nlpcloud'
    AI_ENGINE_OPENAI = 'openai'

    def __nlpcloud_private_callout(client, prompt, temp=0.85):
        return client.generation(
                    prompt,
                    min_length=100,
                    max_length=256,
                    length_no_input=True,
                    remove_input=True,
                    end_sequence=None,
                    top_p=1,
                    temperature=temp,
                    top_k=25,
                    repetition_penalty=1,
                    length_penalty=1,
                    do_sample=True,
                    early_stopping=False,
                    num_beams=1,
                    no_repeat_ngram_size=0,
                    num_return_sequences=1,
                    bad_words=None,
                    remove_end_sequence=False
                    )

    def __openai_private_callout(prompt, engine='text-davinci-002', temp=1, tokens=256, top_p=1.0, freq_pen=0.0, pres_pen=0.0, stop_strings=['zxcv']):
        return openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=tokens,
                    top_p=top_p,
                    frequency_penalty=freq_pen,
                    presence_penalty=pres_pen,
                    stop=stop_strings)

    def nlpcloud_callout(prompt):
        max_retry = 5
        retry = 0
        prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()  # force it to fix any unicode errors
        while True:
            try:
                sleep(1) # Wait 1 second because NLP Cloud will error with HTTP 429 too many requests
                client = nlpcloud.Client(
                    'finetuned-gpt-neox-20b',
                    ZestorHelper.nlp_cloud_api_key,
                    gpu=True,
                    lang='en')

                engine_output = ZestorHelper.__nlpcloud_private_callout(client,prompt)

                text = engine_output['generated_text'].strip()
                text = re.sub('\s+', ' ', text)

                # retry incomplete responses once
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    sleep(1) # Wait 1 second because NLP Cloud will error with HTTP 429 too many requests
                    engine_output = ZestorHelper.__nlpcloud_private_callout(client,prompt+text)

                    text2 = engine_output['generated_text'].strip()
                    text2 = re.sub('\s+', ' ', text2)

                    text = text + ' ' + text2
                
                # retry incomplete responses twice
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    sleep(1) # Wait 1 second because NLP Cloud will error with HTTP 429 too many requests
                    engine_output = ZestorHelper.__nlpcloud_private_callout(client,prompt+text)

                    text2 = engine_output['generated_text'].strip()
                    text2 = re.sub('\s+', ' ', text2)

                    text = text + ' ' + text2

                filename = '%s_nlpcloud.txt' % time()

                ZestorHelper.mkdir_if_not_exists('logs')
                ZestorHelper.mkdir_if_not_exists('logs/nlpcloud')

                ZestorHelper.save_file('logs/nlpcloud/%s' % filename, prompt + '\n\n==========\n\n' + text)
                return text
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return "NLPCLOUD error: %s" % oops
                print('Error communicating with NLP Cloud:', oops)
                sleep(1)

    def openai_callout(prompt, engine='text-davinci-002', temp=1, tokens=256, top_p=1.0, freq_pen=0.0, pres_pen=0.0, stop_strings=['zxcv']):
        max_retry = 5
        retry = 0
        prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
        while True:
            try:
                response = ZestorHelper.__openai_private_callout(prompt, engine, temp, tokens, top_p, freq_pen, pres_pen, stop_strings)
                
                text = response['choices'][0]['text'].strip()
                text = re.sub('\s+', ' ', text)

                # retry incomplete responses once
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    response = ZestorHelper.__openai_private_callout(prompt+text, engine, temp, tokens, top_p, freq_pen, pres_pen, stop_strings)

                    text2 = response['choices'][0]['text'].strip()
                    text2 = re.sub('\s+', ' ', text2)

                    text = text + text2

                # retry incomplete responses twice
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    response = ZestorHelper.__openai_private_callout(prompt+text, engine, temp, tokens, top_p, freq_pen, pres_pen, stop_strings)

                    text2 = response['choices'][0]['text'].strip()
                    text2 = re.sub('\s+', ' ', text2)

                    text = text + text2
                
                filename = '%s_gpt3.txt' % time()

                ZestorHelper.mkdir_if_not_exists('logs')
                ZestorHelper.mkdir_if_not_exists('logs/openai')

                ZestorHelper.save_file('logs/openai/%s' % filename, prompt + '\n\n==========\n\n' + text)
                return text
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return "OpenAI error: %s" % oops
                print('Error communicating with OpenAI:', oops)
                sleep(1)

    def call_ai_engine(AIEngine, prompt):
        scene = ''
        print('\n======================= CALLING AI ENGINE =======================')
        print('\n',prompt)
        if AIEngine == ZestorHelper.AI_ENGINE_OPENAI:
            scene = ZestorHelper.openai_callout(prompt)
        if AIEngine == ZestorHelper.AI_ENGINE_NLPCLOUD:
            scene = ZestorHelper.nlpcloud_callout(prompt)

        print('\n',scene,'\n','=======================')
        return scene

    def cleanup_aiengine_output(text):
        text = re.sub(r'[1-9]+\.\s?', '\r\n', text)
        text = text.replace(': ','-')
        text = os.linesep.join([s for s in text.splitlines() if s])       
        return text

    def only_first_paragraph(this_text):
        retval = ''
        this_text = this_text.strip() # remove spaces
        lines = this_text.splitlines()
        if lines[0]:
            retval = lines[0]
        return retval

    def remove_previous_lines(this_text, previous_scene):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        thisLines = tokenizer.tokenize(this_text)
        previousLines = tokenizer.tokenize(previous_scene)

        textArray = [];
        for thisLine in thisLines:
            lineGood = True
            for previousLine in previousLines:
                if thisLine == previousLine:
                    lineGood = False
            if lineGood:
                textArray.append(thisLine)

        return ' '.join(textArray).strip()

    def get_bestof_longest_text(AIEngine, prompt, previous_scene, numberOfTries):
        # HOW MANY TIMES TO REGEN THE SAME PARAGRAPH
        prompts = []
        for j in range (0,numberOfTries):
            prompts.append(prompt)

        # MULTIPLE SIMULTANEOUS CONCURRENT CALLS TO AI ENGINE
        prompt_queue = []
        with ThreadPoolExecutor(max_workers=numberOfTries) as executor:
            ordinal = 1
            for prompt in prompts:
                prompt_queue.append(executor.submit(ZestorHelper.call_ai_engine, AIEngine, prompt))
                ordinal += 1

        # WAIT FOR ALL SIMULTANEOUS CONCURRENT CALLS TO COMPLETE
        # LOOP TO FIND THE LONGEST PARAGRAPH
        longest_text = ''
        longest_text_length = 0
        for future in concurrent.futures.as_completed(prompt_queue):
            try:
                generated_text = future.result()

                # NLP CLOUD CREATES USUALLY A GOOD FIRST PARAGRAPH, BUT THEN GARBAGE
                if AIEngine == ZestorHelper.AI_ENGINE_NLPCLOUD:
                    generated_text = ZestorHelper.only_first_paragraph(generated_text)

                if generated_text:
                    generated_text = ZestorHelper.remove_previous_lines(generated_text, previous_scene)
                    len_this_generated_text = len(generated_text)
                    if len_this_generated_text > longest_text_length:
                        longest_text_length = len_this_generated_text
                        longest_text = generated_text
                        print('\n=== BEST SO FAR ====> %d size \n%s' % (len_this_generated_text, generated_text))    
                    else:
                        print('\n=== NOT BEST ========> %d size \n%s' % (len_this_generated_text, generated_text))    
                else:
                    print('\n\ngenerated blank')
            except Exception as exc:
                print('\n\ngenerated an exception: %s' % (exc))

        print('\n== CHOSEN LONGEST LENGTH ==> %d size \n%s' % (longest_text_length, longest_text))    

        return longest_text

    # get html source for a url
    def get_source(url):
        try:
            session = HTMLSession()
            response = session.get(url)
            return response

        except requests.exceptions.RequestException as e:
            print(e)

    # first page google links, removing any non-relevant URLs
    def scrape_google(query):

        # https://docs.python.org/3/library/urllib.parse.html
        # URLEncode, replacing special characters with %XX, and space with +
        query = urllib.parse.quote_plus(query)
        response = ZestorHelper.get_source("https://www.google.com/search?q=" + query)

        # remove these type links
        google_domains = ('https://www.google.', 
                            'https://google.', 
                            'https://webcache.googleusercontent.', 
                            'http://webcache.googleusercontent.', 
                            'https://policies.google.',
                            'https://support.google.',
                            'https://maps.google.',
                            'https://example.org',
                            'https://scholar.google.com')

        links = list(response.html.absolute_links)
        for url in links[:]:
            if url.startswith(google_domains):
                links.remove(url)

        return links

    # not used, function to get HTML response object
    def get_results(query):
        
        query = urllib.parse.quote_plus(query)
        response = ZestorHelper.get_source("https://www.google.com/search?q=" + query)
        
        return response

    # not used, function to parse the google page for text, url, and text blurb
    def parse_results(response):
        
        css_identifier_result = ".tF2Cxc"
        css_identifier_title = "h3"
        css_identifier_link = ".yuRUbf a"
        css_identifier_text = ".VwiC3b"
        
        results = response.html.find(css_identifier_result)

        output = []
        
        for result in results:

            item = {
                'title': result.find(css_identifier_title, first=True).text,
                'link': result.find(css_identifier_link, first=True).attrs['href'],
                'text': result.find(css_identifier_text, first=True).text
            }
            
            output.append(item)
            
        return output

    # not used, get raw response and parse google page CSS
    def google_search(query):
        response = ZestorHelper.get_results(query)
        return ZestorHelper.parse_results(response)

    def open_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    def save_file(filepath, content='', mode='w'):
        with open(filepath, mode, encoding='utf-8') as outfile:
            outfile.write(content)

    def mkdir_if_not_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    #====================
    # clean up HTML into text, removing tags, script, style
    def clean_html(html):

        # setup html tag cleaner https://lxml.de/lxmlhtml.html

        tags = ['h1','h2','h3','h4','h5','h6',
            'div', 'span', 
            'img', 'area', 'map']
        args = {'meta':False, 'safe_attrs_only':False, 'page_structure':False, 
            'scripts':True, 'style':True, 'links':True, 'remove_tags':tags}
        cleaner = Cleaner(**args)

        # strip tags, style, script, etc...
        doc = fromstring(html)
        retval = cleaner.clean_html(doc.xpath('/html/body')[0]).text_content() # clean everything in first body tag
        retval = retval.encode('ascii', 'ignore') # encode string to ascii, ignore errors
        retval = retval.decode("utf-8") # decode to UTF-8
        retval = retval.replace('\r',' ') # carriage return
        retval = retval.replace('\n',' ') # new line
        retval = retval.replace('\t',' ') # tab

        # remove special chars, replace doouble space with single space
        retval=re.sub(r'[^\x00-\x7F]',' ', retval)
        
        # loop while retval has double space
        while len(re.findall(r'\s\s', retval)) > 0:
            retval = retval.replace('  ',' ')

        # Natural Language Toolkit line parser https://www.nltk.org/
        # https://www.nltk.org/_modules/nltk/tokenize/punkt.html
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        thisLines = tokenizer.tokenize(retval)

        # loop through lines of text removing unwanted lines
        tempLines = list(thisLines)
        for tempLine in tempLines[:]:
            if not tempLine.endswith(('.','!','?','"')): # any line that doesn't end in punctuation
                tempLines.remove(tempLine)
            elif not re.match('^[a-zA-Z]',tempLine): # any line that doesn't start with alpha
                tempLines.remove(tempLine)
            elif len(tempLine.strip()) < 50: # any line less than 50 characters
                tempLines.remove(tempLine)
            elif len(re.findall('[0-9]+', tempLine)) > 3: # any line with more than 3 occurences of number
                tempLines.remove(tempLine)
            elif len(re.findall(':', tempLine)) > 0: # any line with colon
                tempLines.remove(tempLine)
            elif len(re.findall(';', tempLine)) > 0: # any line with semicolon
                tempLines.remove(tempLine)
            elif len(re.findall('privacy policy', tempLine.lower())) > 0: # any line that says privacy policy
                tempLines.remove(tempLine)
            elif len(re.findall('wikipedia', tempLine.lower())) > 0: # any line with wikipedia
                tempLines.remove(tempLine)
            elif len(re.findall('terms of use', tempLine.lower())) > 0: # any line with terms of use
                tempLines.remove(tempLine)
            elif len(re.findall('terms of service', tempLine.lower())) > 0: # any line with terms of service
                tempLines.remove(tempLine)
        # rejoin all the sentences into a string
        retval = ' '.join(tempLines)
        # return a block of text
        return retval

    #====================
    # recursive relevant summary of large text
    def relevant_summary(SELECTED_AI_ENGINE, text):

        # Natural Language Toolkit line parser https://www.nltk.org/
        # https://www.nltk.org/_modules/nltk/tokenize/punkt.html
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        textLines = tokenizer.tokenize(text)

        retval = '' # additive summary

        # summarize text with AI engine
        tempText = ''
        for line in textLines:
            tempText += line # add a sentence to text
            # when text > 2k chars, summarize
            if len(tempText) > 2000:
                prompt = ZestorHelper.open_file('PromptTemplates/' + SELECTED_AI_ENGINE + '/relevant_summary.txt').replace('<<TEXT>>',tempText)
                retval += ZestorHelper.call_ai_engine(SELECTED_AI_ENGINE, prompt)
                tempText = '' # clear out the temp variable

        # summarize text, when partial text block <= 2k chars at the end
        if len(tempText) > 0 :
            prompt = ZestorHelper.open_file('PromptTemplates/' + SELECTED_AI_ENGINE + '/relevant_summary.txt').replace('<<TEXT>>',tempText)
            retval += ZestorHelper.call_ai_engine(SELECTED_AI_ENGINE, prompt)

        # combined summary
        return retval 

    def get_url_only_text(url):
        response = ZestorHelper.get_source(url)
        responseHtml = response.html.html
        print('\n\n==Response==> %s' % (responseHtml)) 
        clean_response = ZestorHelper.clean_html(responseHtml)
        print('\n\n==Clean Response==> %s' % (clean_response)) 
        return clean_response

    def urldb_getfilename(url):
        ZestorHelper.mkdir_if_not_exists('url_db') # cache directory

        url_db_file = 'url_db/' + url
        url_db_file = url_db_file.replace('https:','_')
        url_db_file = url_db_file.replace('http:','_')
        url_db_file = url_db_file.replace(':','_')
        url_db_file = url_db_file.replace('#','_')
        url_db_file = url_db_file.replace('.','_')
        url_db_file = url_db_file.replace(';','_')
        url_db_file = url_db_file.replace('/','_')
        url_db_file = url_db_file.replace('\\','_')
        url_db_file = url_db_file.replace('(','_')
        url_db_file = url_db_file.replace(')','_')

        # loop while url_db_file has double underscore
        while len(re.findall(r'__', url_db_file)) > 0:
            url_db_file = url_db_file.replace('__','_')

        return url_db_file

    def urldb_exists(url_db_file):
        retval = False
        if exists(url_db_file):
                retval = True
        return retval

    def urldb_open(url_db_file):
        return ZestorHelper.open_file(url_db_file)

    def urldb_save(url_db_file, content):
        ZestorHelper.save_file(url_db_file, content)
