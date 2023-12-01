#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import time
from os.path import exists
from time import time

import nltk.data  # NLP sentence parser used to remove any duplicate word for word sentence output from AI response
import openai  # OpenAI https://www.openai.com
from lxml.html.clean import Cleaner
# https://pypi.org/project/lxml/
# https://lxml.de/lxmlhtml.html
# pip install beautifulsoup4
# https://lxml.de/elementsoup.html
from lxml.html.soupparser import fromstring

# parallelism makes concurrent simulataneous calls to the AI Engine
# From those responses, the best text is determined by length 
# If you want to choose the best of 3 calls for the same next scene, then enter 3 below.

class ZestorHelper:

    open_ai_api_key = os.getenv('OPENAI_API_KEY') # not needed, but for clarity

    AI_ENGINE_OPENAI = 'openai'

    def openai_callout_noretry(prompt, engine='text-davinci-002', temp=1, tokens=256, top_p=1.0, freq_pen=0.0, pres_pen=0.0, stop_strings=['zxcv']):
        max_retry = 1
        retry = 1
        #prompt = prompt.encode(encoding='ASCII',errors='ignore').decode('ASCII',errors='ignore')
        #print(type(prompt))
        while retry <= max_retry:
            try:
                #print('OpenAI callout\r\n%s\r\n' % (prompt))
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=tokens,
                    top_p=top_p,
                    frequency_penalty=freq_pen,
                    presence_penalty=pres_pen,
                    stop=stop_strings)
                #print('OpenAI callout after')
                #print(type(response))
                text = response['choices'][0]['text'].strip()

                #print('OpenAI response #1\r\n%s' % (text))
                text = re.sub('\s+', ' ', text)

                filename = '%s.txt' % time()

                ZestorHelper.mkdir_if_not_exists('logs')
                ZestorHelper.mkdir_if_not_exists('logs/openai')

                ZestorHelper.save_file('logs/openai/%s' % filename, prompt + '\n\n==========\n\n' + text)
                return text
            except Exception as oops:
                retry += 1
                print('Error communicating with OpenAI:', oops)

    def openai_callout(prompt, engine='text-davinci-002', temp=1, tokens=256, top_p=1.0, freq_pen=0.0, pres_pen=0.0, stop_strings=['zxcv']):
        max_retry = 1
        retry = 1
        #prompt = prompt.encode(encoding='ASCII',errors='ignore').decode('ASCII',errors='ignore')
        print(type(prompt))
        while retry <= max_retry:
            try:
                print('OpenAI callout\r\n%s\r\n' % (prompt))
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=tokens,
                    top_p=top_p,
                    frequency_penalty=freq_pen,
                    presence_penalty=pres_pen,
                    stop=stop_strings)
                print('OpenAI callout after')
                print(type(response))
                text = response['choices'][0]['text'].strip()

                print('OpenAI response #1\r\n%s' % (text))
                text = re.sub('\s+', ' ', text)

                # retry incomplete responses once
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    print('OpenAI retrying incomplete prompt #1\r\n%s' % (prompt+text))
                    response = openai.Completion.create(
                        engine=engine,
                        prompt=prompt+text,
                        temperature=temp,
                        max_tokens=tokens,
                        top_p=top_p,
                        frequency_penalty=freq_pen,
                        presence_penalty=pres_pen,
                        stop=stop_strings)
                    text2 = response['choices'][0]['text'].strip()
                    print('OpenAI retrying incomplete response #1\r\n%s' % (text2))
                    text2 = re.sub('\s+', ' ', text2)

                    text = text + text2

                # retry incomplete responses twice
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    print('OpenAI retrying incomplete prompt #2\r\n%s' % (prompt+text))
                    response = openai.Completion.create(
                        engine=engine,
                        prompt=prompt+text,
                        temperature=temp,
                        max_tokens=tokens,
                        top_p=top_p,
                        frequency_penalty=freq_pen,
                        presence_penalty=pres_pen,
                        stop=stop_strings)
                    text2 = response['choices'][0]['text'].strip()
                    print('OpenAI retrying incomplete response #2\r\n%s' % (text2))

                    text2 = re.sub('\s+', ' ', text2)

                    text = text + text2
                
                filename = '%s.txt' % time()

                ZestorHelper.mkdir_if_not_exists('logs')
                ZestorHelper.mkdir_if_not_exists('logs/openai')

                ZestorHelper.save_file('logs/openai/%s' % filename, prompt + '\n\n==========\n\n' + text)
                return text
            except Exception as oops:
                retry += 1
                print('Error communicating with OpenAI:', oops)

        scene = ''
        print('\n======================= CALLING AI ENGINE =======================')
        print('\n',prompt)
        if AIEngine == ZestorHelper.AI_ENGINE_OPENAI:
            scene = ZestorHelper.openai_callout(prompt)
        if AIEngine == ZestorHelper.AI_ENGINE_NLPCLOUD:
            scene = ZestorHelper.nlpcloud_callout(prompt)

        print('\n',scene,'\n','=======================')
        return scene

    def open_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return str(infile.read())

    def save_file(filepath, content='', mode='w'):
        with open(filepath, mode, encoding='utf-8') as outfile:
            outfile.write(content)

    def mkdir_if_not_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    #====================
    # clean up HTML into text, removing tags, script, style
    def clean_html(html):
        retval = ''
        try:
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
        except Exception as oops:
            print('clean_html() Exception: %s' % (oops))
        # return a block of text
        return retval
