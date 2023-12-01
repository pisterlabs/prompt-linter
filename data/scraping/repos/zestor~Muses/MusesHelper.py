#!/usr/bin/python
# -*- coding: utf-8 -*-

import concurrent.futures
import os
import re
from concurrent.futures.thread import ThreadPoolExecutor
from time import sleep, time
import cohere 

import nlpcloud # NLP Cloud Playground https://www.nlpcloud.com
import nltk.data # NLP sentence parser used to remove any duplicate word for word sentence output from AI response
import openai # OpenAI https://www.openai.com

# parallelism makes concurrent simulataneous calls to the AI Engine
# From those responses, the best text is determined by length 
# If you want to choose the best of 3 calls for the same next scene, then enter 3 below.

class MusesHelper:

    open_ai_api_key = os.getenv('OPENAI_API_KEY')
    nlp_cloud_api_key = os.getenv('NLPCLOUD_API_KEY') 
    cohere_api_key = os.getenv('COHERE_PROD_API_KEY')

    AI_ENGINE_NLPCLOUD = 'nlpcloud'
    AI_ENGINE_OPENAI = 'openai'
    AI_ENGINE_COHERE = 'cohere'

    def openFile(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    def saveFile(filepath, content):
        with open(filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(content)

    def mkDirIfNotExists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def __nlpcloudPrivateCallout(client, prompt):
        return client.generation(
                    prompt,
                    min_length=100,
                    max_length=256,
                    length_no_input=True,
                    remove_input=True,
                    end_sequence=None,
                    top_p=1,
                    temperature=0.85,
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

    def __openaiPrivateCallout(prompt):
        return openai.Completion.create(
                    engine='text-davinci-003',
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=2500,
                    top_p=1.0,
                    frequency_penalty=0.3,
                    presence_penalty=0.3,
                    stop=['zxcv'])

    def __coherePrivateCallout(client, prompt):
        return client.generate(
            model='command-xlarge-nightly',
            prompt=prompt,
            max_tokens=750,
            temperature=0.7,
            p=1.0,
            frequency_penalty=0.3,
            presence_penalty=0.3,
            stop_sequences=['zxcv'])

    def nlpcloudCallOut(prompt):
        max_retry = 5
        retry = 0
        prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()  # force it to fix any unicode errors
        while True:
            try:
                sleep(1) # Wait 1 second because NLP Cloud will error with HTTP 429 too many requests
                client = nlpcloud.Client(
                    'finetuned-gpt-neox-20b',
                    MusesHelper.nlp_cloud_api_key,
                    gpu=True,
                    lang='en')

                engine_output = MusesHelper.__nlpcloudPrivateCallout(client,prompt)

                text = engine_output['generated_text'].strip()
                text = re.sub('\s+', ' ', text)

                # retry incomplete responses once
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    sleep(1) # Wait 1 second because NLP Cloud will error with HTTP 429 too many requests
                    engine_output = MusesHelper.__nlpcloudPrivateCallout(client,prompt+text)

                    text2 = engine_output['generated_text'].strip()
                    text2 = re.sub('\s+', ' ', text2)

                    text = text + ' ' + text2
                
                # retry incomplete responses twice
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    sleep(1) # Wait 1 second because NLP Cloud will error with HTTP 429 too many requests
                    engine_output = MusesHelper.__nlpcloudPrivateCallout(client,prompt+text)

                    text2 = engine_output['generated_text'].strip()
                    text2 = re.sub('\s+', ' ', text2)

                    text = text + ' ' + text2

                filename = '%s_nlpcloud.txt' % time()

                MusesHelper.mkDirIfNotExists('logs')
                MusesHelper.mkDirIfNotExists('logs/nlpcloud')

                MusesHelper.saveFile('logs/nlpcloud/%s' % filename, prompt + '\n\n==========\n\n' + text)
                return text
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return "NLPCLOUD error: %s" % oops
                print('Error communicating with NLP Cloud:', oops)
                sleep(1)

    def openaiCallOut(prompt):
        max_retry = 5
        retry = 0
        prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
        while True:
            try:
                response = MusesHelper.__openaiPrivateCallout(prompt)
                
                text = response['choices'][0]['text'].strip()
                text = re.sub('\s+', ' ', text)

                # retry incomplete responses once
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    response = MusesHelper.__openaiPrivateCallout(prompt+text)

                    text2 = response['choices'][0]['text'].strip()
                    text2 = re.sub('\s+', ' ', text2)

                    text = text + text2

                # retry incomplete responses twice
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    response = MusesHelper.__openaiPrivateCallout(prompt+text)

                    text2 = response['choices'][0]['text'].strip()
                    text2 = re.sub('\s+', ' ', text2)

                    text = text + text2
                
                filename = '%s_gpt3.txt' % time()

                MusesHelper.mkDirIfNotExists('logs')
                MusesHelper.mkDirIfNotExists('logs/openai')

                MusesHelper.saveFile('logs/openai/%s' % filename, prompt + '\n\n==========\n\n' + text)
                return text
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return "OpenAI error: %s" % oops
                print('Error communicating with OpenAI:', oops)
                sleep(1)

    def cohereCallOut(prompt):
        max_retry = 5
        retry = 0
        prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
        while True:
            try:
                client = cohere.Client(MusesHelper.cohere_api_key)

                response = MusesHelper.__coherePrivateCallout(client, prompt)
                
                text = response.generations[0].text.strip()
                text = text.replace('"/','"')
                text = re.sub('\s+', ' ', text)

                # retry incomplete responses once
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    response = MusesHelper.__coherePrivateCallout(client, prompt+text)

                    text2 = response.generations[0].text.strip()
                    text2 = text2.replace('"/','"')
                    text2 = re.sub('\s+', ' ', text2)

                    text = text + text2

                # retry incomplete responses twice
                # last character is not some type of sentence ending punctuation
                if not text.endswith(('.','!','?','"')):
                    response = MusesHelper.__coherePrivateCallout(client, prompt+text)

                    text2 = response.generations[0].text.strip()
                    text2 = text2.replace('"/','"')
                    text2 = re.sub('\s+', ' ', text2)

                    text = text + text2
                
                filename = '%s_cohere.txt' % time()

                MusesHelper.mkDirIfNotExists('logs')
                MusesHelper.mkDirIfNotExists('logs/cohere')

                MusesHelper.saveFile('logs/cohere/%s' % filename, prompt + '\n\n==========\n\n' + text)
                return text
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return "COHERE error: %s" % oops
                print('Error communicating with CO:HERE:', oops)
                sleep(1)

    def callAIEngine(AIEngine, prompt):
        scene = ''
        print('\n======================= CALLING AI ENGINE =======================')
        print('\n',prompt)
        if AIEngine == MusesHelper.AI_ENGINE_OPENAI:
            scene = MusesHelper.openaiCallOut(prompt)
        if AIEngine == MusesHelper.AI_ENGINE_NLPCLOUD:
            scene = MusesHelper.nlpcloudCallOut(prompt)
        if AIEngine == MusesHelper.AI_ENGINE_COHERE:
            scene = MusesHelper.cohereCallOut(prompt)

        print('\n',scene,'\n','=======================')
        return scene

    def cleanUpAIengineOutput(text):
        text = re.sub(r'[1-9]+\.\s?', '\r\n', text)
        text = text.replace(': ','-')
        text = os.linesep.join([s for s in text.splitlines() if s])       
        return text

    def OnlyFirstParagraph(this_text):
        retval = ''
        this_text = this_text.strip() # remove spaces
        lines = this_text.splitlines()
        if lines[0]:
            retval = lines[0]
        return retval

    def removeAnyPreviousLines(this_text, previous_scene):
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

    def getLongestText(AIEngine, prompt, previous_scene, numberOfTries):
        # HOW MANY TIMES TO REGEN THE SAME PARAGRAPH
        prompts = []
        for j in range (0,numberOfTries):
            prompts.append(prompt)

        # MULTIPLE SIMULTANEOUS CONCURRENT CALLS TO AI ENGINE
        prompt_queue = []
        with ThreadPoolExecutor(max_workers=numberOfTries) as executor:
            ordinal = 1
            for prompt in prompts:
                prompt_queue.append(executor.submit(MusesHelper.callAIEngine, AIEngine, prompt))
                ordinal += 1

        # WAIT FOR ALL SIMULTANEOUS CONCURRENT CALLS TO COMPLETE
        # LOOP TO FIND THE LONGEST PARAGRAPH
        longest_text = ''
        longest_text_length = 0
        for future in concurrent.futures.as_completed(prompt_queue):
            try:
                generated_text = future.result()

                # NLP CLOUD CREATES USUALLY A GOOD FIRST PARAGRAPH, BUT THEN GARBAGE
                if AIEngine == MusesHelper.AI_ENGINE_NLPCLOUD:
                    generated_text = MusesHelper.OnlyFirstParagraph(generated_text)

                if generated_text:
                    generated_text = MusesHelper.removeAnyPreviousLines(generated_text, previous_scene)
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