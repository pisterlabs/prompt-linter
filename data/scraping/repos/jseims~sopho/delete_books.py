#!/usr/bin/env python

import requests
import sys
import time
import db
import openai
import hashlib
import random

OPENAI_API_KEY = ''

try:
  from localsettings import *
except:
  print("Error reading localsettings")


def dictValuePad(key):
    return '%(' + str(key) + ')s'

def insertFromDict(table, dict):
    """Take dictionary object dict and produce sql for 
    inserting it into the named table"""
    sql = 'INSERT INTO ' + table
    sql += ' ('
    sql += ', '.join(dict)
    sql += ') VALUES ('
    sql += ', '.join(map(dictValuePad, dict))
    sql += ');'
    return sql


def main(argv):
    book_ids = [8, 14, 18, 20]

    for book_id in book_ids:
        prompt_responses = list(db.query("""SELECT id FROM prompt_response WHERE book_id = %s""", [book_id]))

        print("Found %s prompt responses" % len(prompt_responses))

        for pr in prompt_responses:
            #response_pieces = list(db.query("""SELECT id FROM response_piece WHERE prompt_response_id = %s""", [pr['id']]))
            #print("\tFound %s response pieces" % len(response_pieces))
            db.query("""DELETE FROM response_piece WHERE prompt_response_id = %s""", [pr['id']])
        
        db.query("""DELETE FROM prompt_response WHERE book_id = %s""", [book_id])
        db.query("""DELETE FROM book WHERE id = %s""", [book_id])

if __name__ == "__main__":
   main(sys.argv[1:])
