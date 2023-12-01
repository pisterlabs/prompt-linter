from dotenv import load_dotenv
from langchain.graphs.networkx_graph import NetworkxEntityGraph
import argparse
import ast
import os
import numpy as np
import pandas as pd
import pandasql as ps
import re

def readTriplesFromFile(filePath: str) -> pd.DataFrame:
  data = []
  with open(filePath, "r") as f:
    book = ""
    if re.search("game_of_thrones", filePath):
      book = "Game Of Thrones"
    elif re.search("a_clash_of_kings", filePath):
      book = "A Clash Of Kings"
    elif re.search("a_storm_of_swords", filePath):
      book = "A Storm Of Swords"
    elif re.search("a_feast_for_crows", filePath):
      book = "A Feast For Crows"
    elif re.search("a_dance_with_dragons", filePath):
      book = "A Dance With Dragons"

    i = 0

    for l in f.readlines():
      i += 1
      if i == 1:
        continue
      line = l.split(":", 1)

      print(line)

      page = line[0].strip()
      triples = ast.literal_eval(line[1].strip())
      for triple in triples:
        subject = triple[0].strip()
        object = triple[1].strip()
        predicate = triple[2].strip()
        data.append([book, page, subject, predicate, object])

  df = pd.DataFrame(data, columns=["Book", "Page", "Subject", "Predicate", "Object"])   
  return df

def saveTriplesToFile(df: pd.DataFrame, filePath: str):
  df.to_csv(filePath, sep = "|", index=False)

def readTriplesFromDfFile(filePath: str) -> pd.DataFrame:
  df = pd.read_csv(filePath, sep = "|") 
  return df

#####################################################################################
# Run a SQL statement against the dataframe
#####################################################################################
def runSql(df: pd.DataFrame, sql: str) -> pd.DataFrame:
  return ps.sqldf(sql, locals())

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--file", type=str, required=True, help="Specify the filename where dataframe was saved")
  parser.add_argument("-q", "--sql", type=str, required=True, help="Specify the SQL statement")
  args = parser.parse_args()

  pd.options.display.max_rows = 100
  df = readTriplesFromDfFile(filePath = args.file)

  dfQuery = runSql(df = df, sql = args.sql)
  saveTriplesToFile(df = dfQuery, filePath = "output.csv")
  print(dfQuery) 
