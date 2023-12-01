"""
This uses v2 api
https://learn.microsoft.com/en-us/azure/azure-functions/functions-reference-python?tabs=asgi%2Capplication-level&pivots=python-mode-decorators
"""

import datetime
import azure.functions as func
import logging
import os
import time
import json
from lib.summarydb import SummaryDB

from lib.newsgpt import NewsGPT, NewsCategory, NewsLength
from lib.news_scrapper import get_content, get_news

from azure.cosmos import exceptions, CosmosClient, PartitionKey
from populate_db import populate_db

# cosmos db impl
# https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/cosmos/azure-cosmos/samples/examples.py

app = func.FunctionApp()

##############################################################################


@app.function_name(name="get_last_update")
@app.route(route="last_update", auth_level=func.AuthLevel.ANONYMOUS)
def get_last_update(req: func.HttpRequest) -> func.HttpResponse:
    db = SummaryDB()
    ts = db.query_overall_latest_summary()['_ts']
    return func.HttpResponse(str(ts))

##############################################################################

# TODO: remove this endpoint and the html file
@app.function_name(name="get_static")
@app.route(route="index", auth_level=func.AuthLevel.ANONYMOUS)
def get_static(req: func.HttpRequest) -> func.HttpResponse:
    f = open(os.path.dirname(os.path.realpath(__file__)) + '/index/index.html')
    return func.HttpResponse(f.read(), mimetype='text/html')

##############################################################################


@app.function_name(name="submit_news_form")
@app.route(route="submit_news_form", auth_level=func.AuthLevel.ANONYMOUS)
def submit_news_form(
    req: func.HttpRequest,
) -> func.HttpResponse:
    # receives a POST request with a JSON body and return a json response
    req_body = req.get_json()

    # json_response
    json_res = {"verbose": "Got response from the request."}

    if req_body:
        db = SummaryDB()
        cat = req_body['category']
        len = req_body['length']
        
        len_map = {
            'short': NewsLength.SHORT,
            'medium': NewsLength.MEDIUM,
            'long': NewsLength.LONG
        }

        s = db.query_latest_summary(cat, len_map[len])
        json_res["headlines"] = s["summary"]

        return func.HttpResponse(
            body=json.dumps(json_res),
            status_code=200
        )
    else:
        json_res["verbose"] = "Error: No JSON body found in the request."
        return func.HttpResponse(
            body=json.dumps(json_res),
            status_code=400
        )

##############################################################################
# Cron job to update the database from langchain
# https://learn.microsoft.com/en-us/azure/azure-functions/functions-bindings-timer?tabs=python-v2%2Cin-process&pivots=programming-language-python


# NOTE: local test with `func start` does not work with cron job 
# that is more than 1 min, dunno why
@app.function_name(name="mytimer")
@app.schedule(# cron with 6 fields, for every 10 minutes
            #   schedule="0 */10 * * * *",
              
              # cron with 6 fields, for every hour
            #   schedule="0 0 * * * *",

              # cron with 6 fields, for every 3 hours
              schedule="0 0 */3 * * *",

              arg_name="mytimer",
              run_on_startup=False) 
def test_function(mytimer: func.TimerRequest) -> None:
    logging.info('Python timer trigger function')
    
    # seleluim is not working on azure function
    populate_db(use_seleluim=False)
