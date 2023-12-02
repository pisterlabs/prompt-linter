# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import json
import azure.functions as func
import openai
import re
import requests
import sys
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import (
        TextAnalyticsClient,
        ExtractSummaryAction
    ) 

opanaiKey = os.environ['OpenAiKey']
openaiEndpoint = os.environ['OpenAiEndPoint']
openAiVersion = os.environ['OpenAiVersion']
languageKey = os.environ['LanguageKey']
languageEndpoint = os.environ['LanguageEndPoint']

#Splits text after sentences ending in a period. Combines n sentences per chunk.
def splitter(n, s):
    pieces = s.split(". ")
    list_out = [" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n)]
    return list_out

# Perform light data cleaning (removing redudant whitespace and cleaning up punctuation)
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info(f'{context.function_name} HTTP trigger function processed a request.')
    if hasattr(context, 'retry_context'):
        logging.info(f'Current retry count: {context.retry_context.retry_count}')
        
        if context.retry_context.retry_count == context.retry_context.max_retry_count:
            logging.info(
                f"Max retries of {context.retry_context.max_retry_count} for "
                f"function {context.function_name} has been reached")

    try:
        userQuery = req.params.get('userQuery')
        totalDocs = req.params.get('totalDocs')
        modelName = req.params.get('modelName')
        modelType = req.params.get('modelType')
        logging.info("Input parameters : " + userQuery + " " + totalDocs + " " + modelName + " " + modelType)
        body = json.dumps(req.get_json())
    except ValueError:
        return func.HttpResponse(
             "Invalid body",
             status_code=400
        )
    
    if body:
        result = compose_response(userQuery, totalDocs, modelName, modelType, body)
        return func.HttpResponse(result, mimetype="application/json")
    else:
        return func.HttpResponse(
             "Invalid body",
             status_code=400
        )

def compose_response(userQuery, totalDocs, modelName, modelType, json_data):
    values = json.loads(json_data)['values']
    
    logging.info("Calling Compose Response")
    # Prepare the Output before the loop
    results = {}
    results["values"] = []

    for value in values:
        output_record = transform_value(value, userQuery, totalDocs, modelName, modelType)
        if output_record != None:
            results["values"].append(output_record)
    return json.dumps(results, ensure_ascii=False)        

def summarizeOpenAi(userQuery, myStringList, modelName):
    logging.info("Calling Summarize Open AI")
    openai.api_type = "azure"
    openai.api_key = opanaiKey
    openai.api_base = openaiEndpoint
    openai.api_version = openAiVersion

    '''
    Designing a prompt that will show and tell GPT-3 how to proceed. 
    + Providing an instruction to summarize the text about the general topic (prefix)
    + Providing quality data for the chunks to summarize and specifically mentioning they are the text provided (context + context primer)
    + Providing a space for GPT-3 to fill in the summary to follow the format (suffix)
    '''

    #prompt_i = userQuery + '\n\n\Text:\n' + ' '.join([normalize_text(myStringList)])
    prompt_i = userQuery + '\m\nText:\n' + ' '.join(myStringList)
    #logging.info(prompt_i)

    # for item in myStringList:
    #     logging.info(prompt_i)
    #     logging.info(item)
    #     prompt_i = prompt_i + '\n\n\Text:\n' + " ".join([normalize_text(item)])

    prompt = "".join([prompt_i, '\n\n Summary:\n'])

    #logging.info("Prompt ", prompt)

    # Using a temperature a low temperature to limit the creativity in the response. 
    response = openai.Completion.create(
            engine= modelName,
            prompt = prompt,
            temperature = 0.4,
            max_tokens = 500,
            top_p = 1.0,
            frequency_penalty=0.5,
            presence_penalty = 0.5,
            best_of = 1
        )

    summaryResponse = response.choices[0].text
    return summaryResponse

def summarizeLanguage(myStringList):
    logging.info("Calling Summarize Language")
    text_analytics_client = TextAnalyticsClient(
        endpoint=languageEndpoint,
        credential=AzureKeyCredential(languageKey),
    )
    document = []
    document.append(myStringList)
    print(myStringList)
    # document = [
    #     "At Microsoft, we have been on a quest to advance AI beyond existing techniques, by taking a more holistic, "
    #     "human-centric approach to learning and understanding. As Chief Technology Officer of Azure AI Cognitive "
    #     "Services, I have been working with a team of amazing scientists and engineers to turn this quest into a "
    #     "reality. In my role, I enjoy a unique perspective in viewing the relationship among three attributes of "
    #     "human cognition: monolingual text (X), audio or visual sensory signals, (Y) and multilingual (Z). At the "
    #     "intersection of all three, there's magic-what we call XYZ-code as illustrated in Figure 1-a joint "
    #     "representation to create more powerful AI that can speak, hear, see, and understand humans better. "
    #     "We believe XYZ-code will enable us to fulfill our long-term vision: cross-domain transfer learning, "
    #     "spanning modalities and languages. The goal is to have pretrained models that can jointly learn "
    #     "representations to support a broad range of downstream AI tasks, much in the way humans do today. "
    #     "Over the past five years, we have achieved human performance on benchmarks in conversational speech "
    #     "recognition, machine translation, conversational question answering, machine reading comprehension, "
    #     "and image captioning. These five breakthroughs provided us with strong signals toward our more ambitious "
    #     "aspiration to produce a leap in AI capabilities, achieving multisensory and multilingual learning that "
    #     "is closer in line with how humans learn and understand. I believe the joint XYZ-code is a foundational "
    #     "component of this aspiration, if grounded with external knowledge sources in the downstream AI tasks."
    # ]
    try:
        
        poller = text_analytics_client.begin_analyze_actions(
            document,
            actions=[ExtractSummaryAction(max_sentence_count=4)],
        )

        document_results = poller.result()
        for result in document_results:
            extract_summary_result = result[0]  # first document, first result
            if extract_summary_result.is_error:
                summaryResponse = "...Is an error with code '{}' and message '{}'".format(
                    extract_summary_result.code, extract_summary_result.message)
            else:
                summaryResponse = " ".join([sentence.text for sentence in extract_summary_result.sentences])
    except:
        print("Exception occured")
        summaryResponse = "Exception occured"

    return summaryResponse

## Perform an operation on a record
def transform_value(record, userQuery, totalDocs, modelName, modelType):
    logging.info("Calling Tranform Value")
    try:
        recordId = record['recordId']
    except AssertionError  as error:
        return None

    # Validate the inputs
    try:
        assert ('data' in record), "'data' field is required."
        data = record['data']        
        assert ('text' in data), "'text' field is required in 'data' object."   

    except KeyError as error:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "KeyError:" + error.args[0] }   ]       
            })
    except AssertionError as error:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "AssertionError:" + error.args[0] }   ]       
            })
    except SystemError as error:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "SystemError:" + error.args[0] }   ]       
            })

    try:
        # Getting the items from the values/data/text
        myStringList = []
        myStringList = data['text']

        # Cleaning the list, removing duplicates
        openAiList = list(dict.fromkeys(myStringList))

        if modelType == "Language":
            summaryResponse = summarizeLanguage(myStringList)
        elif modelType == "OpenAI":
            summaryResponse = summarizeOpenAi(userQuery, openAiList, modelName)

        
        return ({
            "recordId": recordId,
            "data": {
                "text": summaryResponse
                    }
            })

    except:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "Could not complete operation for record." }   ]
            })