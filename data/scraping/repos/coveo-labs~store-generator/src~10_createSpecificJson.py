import boto3
import sys
import os
import json
import traceback
import re
import logging
from botocore.exceptions import ClientError
import base64
import operator
import math
import openai
# generate random integer values
from random import seed
from random import randint
import glob

DIRECTORY='sfdc'
PRODUCT_LIST=[{'product':'Guidare Boat Navigation Software Suite','id':'01t8c00000LeoFpAAJ','name':'Guidare'},
              {'product':'Assistente Boat Assisted Software Suite','id':'01t8c00000LeoFtAAJ','name':'Assistente'},
              {'product':'Vedere Vessel Management Software','id':'01t8c00000LeoFoAAJ', 'name':'Vedere'},
              ]
OWNER_LIST=[{'id':'0058c00000BUmI9AAL','name':'Peter Robberts'},
              {'id':'0058c00000BUmIEAA1','name':'Amanda Smith'},
              {'id':'0058c00000BUmIFAA1', 'name':'Jane Jones'},
              {'id':'0058c00000BUmIGAA1', 'name':'John Williams'},
              {'id':'0058c00000BUmIHAA1', 'name':'James Miller'},
              {'id':'0058c00000BUmIIAA1', 'name':'Richard Davis'},
              {'id':'0058c00000BUmIJAA1', 'name':'Thomas Brown'},
              {'id':'0058c00000BUmIKAA1', 'name':'Mark Garcia'},
              ]


def getProductId(jsondata):
  prodId = ''
  product=jsondata['product']
  for prod in PRODUCT_LIST:
    if prod['product']==product:
        prodId = prod['id']
  return prodId

def getOwnerId():
  ownerId = ''
  occurence = randint(0, len(OWNER_LIST)-1)
  return OWNER_LIST[occurence]


def transformToSFDCCase(jsondata):
  #input:
  #   {
  #   "filename": "CASE_Mercury_1_activating the assistent",
  #   "case": "C1004066",
  #   "title": "On my Mercury, how do i fix: activating the assistent is shutting down the Assistente Boat Assisted Software Suite application.",
  #   "boat": "Mercury",
  #   "version": "1",
  #   "product": "Vedere"
  #   "problem": "On my Mercury, how do i fix: activating the assistent is shutting down the Assistente Boat Assisted Software Suite application.",
  #   "symptoms": "The Mercury Assistent boat assisted software suite was shut down because the manual activation process.",
  #   "comments": {
  #     "filename": "CASECOMM_Mercury_1_C1004066",
  #     "comment": "Actually, Mercury's Assist software suite was shut down because it had not been properly updated in some time and its proprietary manual activation process no longer functioned. The Assist software suite is now available through the Mercury Service Dashboard.",
  #     "case": "C1004066"
  #   }
  # }
  ######################################
  #output:
  # {
  #           "attributes": {
  #               "type": "Case",
  #               "referenceId": "CaseRef1"
  #           },
  #           "Subject": "charges button is not working",
  #           "Origin": "Web",
  #           "Reason": "Equipment Complexity",
  #           "Description": "My charges button is not working (couldn't prees it) but I found out that I could tap the Speedbit and it will still working but then I had trouble scanning and I would change something in the settings and it wouldn't change so I decided to make a new account so I deleted the old one but now I need to hold the button for 3 seconds but I can't can someone plz help",
  #           "Priority": "Medium",
  #           "Status": "New",
  #           "Type": "Electrical"
  #"AccountId": "@AccountRef3"
  #       }
  newjson={}
  newjson["attributes"]={}
  newjson["attributes"]["type"]="Case"
  newjson["attributes"]["referenceId"]=jsondata["case"]
  newjson["Subject"]=jsondata["title"]
  newjson["ContactId"]=''+jsondata["name"]
  newjson["AccountId"]=''+jsondata["account"]
  owner = getOwnerId()
  newjson["OwnerId"]=owner['id']
  newjson["OwnerName"]=owner['name']
  newjson["ProductId"]=''+getProductId(jsondata)
  newjson["Origin"]="Web"
  newjson["Reason"]="Setup"
  newjson["Description"]=jsondata["symptoms"]
  newjson["Priority"]="Medium"
  newjson["Status"]="New"
  newjson["Type"]="Problem"
  newjson["SuppliedName"]="Generator"
  return newjson

def transformToSFDCKB(jsondata):
  #https://help.salesforce.com/s/articleView?id=knowledge_article_importer.htm&language=en_US&type=5
  # input:
  # {
  # "filename": "IS_Mercury_1_a blank screen",
  # "title": "How to solve a blank screen for Vedere Vessel Management Software on a Mercury.",
  # "boat": "Mercury",
  # "version": "1",
  # "problem": "The Vedere Vessel Management Software may have lost power and crashed, resulting in a blank screen.",
  # "symptoms": "The symptoms for this problem may include: a blank screen, no sound, or freezing.",
  # "instructions": "The Vedere Vessel Management Software may have lost power and crashed, resulting in a blank screen. <BR>The user can reinstall the software from the manufacturer's website."
  # }
  # ouput:
  # "Title", "summary__c", "Record Type":"FAQ", ""
  return jsondata

def transformToSFDCComment(jsondata):
  #input:
  #   {
  #   "filename": "CASE_Mercury_1_activating the assistent",
  #   "case": "C1004066",
  #   "title": "On my Mercury, how do i fix: activating the assistent is shutting down the Assistente Boat Assisted Software Suite application.",
  #   "boat": "Mercury",
  #   "version": "1",
  #   "problem": "On my Mercury, how do i fix: activating the assistent is shutting down the Assistente Boat Assisted Software Suite application.",
  #   "symptoms": "The Mercury Assistent boat assisted software suite was shut down because the manual activation process.",
  #   "comments": {
  #     "filename": "CASECOMM_Mercury_1_C1004066",
  #     "comment": "Actually, Mercury's Assist software suite was shut down because it had not been properly updated in some time and its proprietary manual activation process no longer functioned. The Assist software suite is now available through the Mercury Service Dashboard.",
  #     "case": "C1004066"
  #   }
  # }
  ######################################
  #output:
  # {
  #           "attributes": {
  #               "type": "CaseComment",
  #               "referenceId": "CaseCommentRef1"
  #           },
  #           "CommentBody": "SteveH Wrote : If the button is broken then the only option would be to replace the Speedbit as it tends to not be repairable.\r\n \r\nIf you are still within warranty (see www.fitbit.com/returns ) then you can start the process by contacting customer support via one of the options in this link:",
  #           "ParentId": "@CaseRef1"
  #       }
  newjson={}
  newjson["attributes"]={}
  newjson["attributes"]["type"]="CaseComment"
  newjson["attributes"]["referenceId"]=jsondata["case"]+"1"
  newjson["CommentBody"]=jsondata["comments"]["comment"].replace('<BR>',"")
  newjson["ParentId"]='@'+jsondata["case"]
  #newjson["CreatorName"]=''+jsondata["CreatorName"]
  #https://salesforce.stackexchange.com/questions/245214/how-to-specify-comment-case-owner
  return newjson


def loadJson(filename):
  newrecord={}
  with open(filename, "r",encoding='utf-8') as fh:
        text = fh.read()
        newrecord = json.loads(text)
  return newrecord

def save(nr,cases, comments):
  #save the files
  plan = [
  {
    "sobject": "Case",
    "saveRefs": True,
    "resolveRefs": False,
    "files": [
      str(nr)+"Cases.json"
    ]
  },
  {
    "sobject": "CaseComment",
    "saveRefs": False,
    "resolveRefs": True,
    "files": [
      str(nr)+"CasesComments.json"
    ]
  }
]
  with open(DIRECTORY+"\\"+str(nr)+"Plan.json", "w", encoding='utf-8') as handler:
        text = json.dumps(plan, ensure_ascii=True)
        handler.write(text)

  with open(DIRECTORY+"\\"+str(nr)+"Cases.json", "w", encoding='utf-8') as handler:
        jsondata={}
        jsondata["records"]=cases
        text = json.dumps(jsondata, ensure_ascii=True)
        handler.write(text)
  with open(DIRECTORY+"\\"+str(nr)+"CasesComments.json", "w", encoding='utf-8') as handler:
        jsondata={}
        jsondata["records"]=comments
        text = json.dumps(jsondata, ensure_ascii=True)
        handler.write(text)

def process(filename):
  #Cannot contain more than 200 records...
  #So we need to split them up
  total=0
  nr = 1
  alljsoncases=[]
  alljsoncomments=[]
  jsons = glob.glob('json\\CASE_*.json',recursive=True)
  for filename in jsons:
    print("Processing: "+filename)
    jsondata = loadJson(filename)
    case = transformToSFDCCase(jsondata)
    jsondata['CreatorName']=case['OwnerName']
    del case['OwnerName']
    casecomment = transformToSFDCComment(jsondata)
    alljsoncases.append(case)
    alljsoncomments.append(casecomment)
    total+=2
    if (total>150):
      save(nr, alljsoncases, alljsoncomments)
      nr+=1
      total = 0
      alljsoncases=[]
      alljsoncomments=[]


  
  
  print("We are done!\n")
  print("Processed: "+str(total)+" results\n")


def processKB(filename):
  total=0
  alljsonkbs=[]
  jsons = glob.glob('json\\IS_*.json',recursive=True)
  for filename in jsons:
    print("Processing: "+filename)
    jsondata = loadJson(filename)
    kb = transformToSFDCKB(jsondata)
    alljsonkbs.append(case)
    total+=1
  #save the files
  with open(DIRECTORY+"\\KBs.json", "w", encoding='utf-8') as handler:
        jsondata={}
        jsondata["records"]=alljsonkbs
        text = json.dumps(jsondata, ensure_ascii=True)
        handler.write(text)
  
  print("We are done!\n")
  print("Processed: "+str(total)+" results\n")  

try:
  #fileconfig = sys.argv[1]
  #process(fileconfig)
  process('')
  #processKB('')
except Exception as e: 
  print(e)
  traceback.print_exception(*sys.exc_info())
  #print ("Specify configuration json (like config.json) on startup")
