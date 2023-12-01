# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""This module is for running the db agent use case."""
import urllib
import warnings
warnings.filterwarnings("ignore")

import os
from langchain import SQLDatabase, SQLDatabaseChain, OpenAI

import connect, models

dsn = os.getenv("DSN")
database = os.getenv("DATABASE")

quoted = urllib.parse.quote_plus(f'DSN={dsn};DATABASE={database}')

connection_string = f"mssql+pyodbc:///?odbc_connect={quoted}"


def run_db_agent(url, llm, query):
    db = SQLDatabase.from_uri(url)
    db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)
    response = db_chain.run(query)
    return response


if __name__ == "__main__":
    llm = models.set_openai_model(temperature=0)
    query = "How many rooms were rented in Manhattan from January to June 2021?"
    response = run_db_agent(url=connection_string, llm=llm, query=query)
    print(response)