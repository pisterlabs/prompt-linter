""" MIT License

Copyright (c) 2023 Institute for Quantitative Social Science, Stefano M. Iacus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 """

# requirements.txt
# python=3.10
# Jinja2
# langchain
# openai
# shiny
# pandas

# This code, my first python code, was created in a night. Be merciful.

# This script requires two arguments: `fileId` and `siteUrl`. 
# Note: The url arguments are case sensitive.
# The parameter `datasetPid` is collected but not yet used
# for Harvard Dataverse siteUrl always equals to https://dataverse.harvard.edu
# fileId is the internal identifier in the Dataverse database
# example of use:
# https://askdataverse.shinyapps.io/askthedata/?fileId=4862482siteUrl=https://dataverse.harvard.edu
# or, if run locally
# http://localhost:64504/?fileId=4862482&siteUrl=https://dataverse.harvard.edu
# replace 64504 with your port

# AskTheData

from shiny import  App, reactive, render, ui
import asyncio
import re
from pathlib import Path
import pandas as pd
import requests   # for urls 
import io
import sqlite3
import os
from htmltools import HTML, div
import string
import random
import duckdb
import duckdb_engine

from urllib.request import urlopen
import json
import urllib.parse
from urllib.parse import urlparse, parse_qs

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor


mypath = "./"
os.chdir(mypath)

OPENAI_API_KEY = 'xxxxxxxxx' # your OpenAI key

# We use OpenAI `text-davinci-003` but it can be changed with other models supported by LangChain
myllm = llm=OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY, model_name = "text-davinci-003")

# these variables must remain global
apiStr = '/api/access/datafile/'
fileid = ""
dataset_pid = ""
dataurl = ""
siteUrl = ""


def app_ui(request):
    global fileid
    global dataset_pid
    global dataurl
    global apiStr

    fileid = request.query_params.get('fileId')
    dataset_pid = request.query_params.get('datasetPid')
    siteUrl = request.query_params.get('siteUrl')
    if fileid is None:
        fileid = ''
    if siteUrl is None:
        siteUrl = ''
    dataurl = siteUrl + apiStr + fileid
    
    _ui = ui.page_fluid(
        ui.input_text_area("query", "Tell me what you want to know", placeholder="What is this data about?"),
        ui.input_action_button("think", "Answer please", class_="btn-primary"),
        ui.output_text("answer"),
        ui.output_data_frame("grid")
    )
    return _ui

# some variables need to leave within server() so they remain local to the user session
# otherwise two users may load different data that overlaps in the shiny session

def server(input, output, session):
    HaveData = reactive.Value(None)
    HaveQuery = reactive.Value(None)
    sqlDB = reactive.Value(None)

    @reactive.Calc
    async def load_tabular_data():
        print('Loading data')
        HaveData.set(False)
        ui.notification_show("Loading data...", id='loadingID', duration=None, type='warning')
        req = requests.get(dataurl).content
        data = pd.read_csv(io.StringIO(req.decode('utf-8')), sep=None, engine="python")
        df = pd.DataFrame(data)
        tmpDB = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=10))
        tmpDB = tmpDB + '.db'
        db = os.path.join(mypath, tmpDB)
        sqlDB.set(db)       
        if os.path.exists(db):
            os.remove(db)
        con = duckdb.connect(db)
        con.execute('CREATE TABLE data AS SELECT * FROM df')
        HaveData.set(True) 
        ui.notification_remove('loadingID')    
        return df
   
    @output
    @render.data_frame
    async def grid():
        data = await load_tabular_data()
        return render.DataGrid(data,height=350,width="fit-content")

    @output
    @render.text
    async def answer():
        input.think()
        with reactive.isolate():
            ans = "Waiting for you..."
            if(HaveData.get()):
                mydb = SQLDatabase.from_uri(  "duckdb:///" + sqlDB.get() )
                toolkit = SQLDatabaseToolkit(db=mydb,llm=myllm)
                if HaveQuery.get():
                    this_query = input.query()
                else:
                    this_query = 'What is this data about?' # default initial query
                    HaveQuery.set(True)    # we need this here
                ui.notification_show("Thinking...", id='thinkingID', duration=None)    
                agent_executor = create_sql_agent(
                    llm=myllm,toolkit=toolkit,verbose=True)   
                ans = agent_executor.run(f"{this_query}")
                ui.notification_remove('thinkingID')    
        return f"{ans}"    
        
app = App(app_ui, server)

