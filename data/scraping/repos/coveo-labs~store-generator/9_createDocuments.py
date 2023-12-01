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
import csv

import glob

P_ENGINE = 'text-curie-001'
FILECOUNTER=2000
FILENAME=''
FILENAME_ACCOUNTS='AccountsAndContacts.csv'
CASES_ONLY = True
ALL_ACCOUNTS=[]
ACCOUNT_POINTER=0

def readCSV():
  global ALL_ACCOUNTS
  with open(FILENAME_ACCOUNTS, mode='r') as csv_file:
    ALL_ACCOUNTS = list(csv.DictReader(csv_file, delimiter=';'))

def cleanUp(text):
  text = text.replace('<|endoftext|>','')
  text = text.replace('\n','').replace('* * *','').strip()
  return text

def removeUnfinishedSentence(text):
  #text bla. bla bla --> remove the last part
  if (not text.endswith('.')):
    if ('.' in text):
    	text = '.'.join(text.split('.')[:-1])
  return text


def executeOpenAI(prompt, temp, length, stop=[]):
  if len(stop)==0:
    stop=None
  results = openai.Completion.create(
    engine=P_ENGINE,
    prompt=prompt,
    temperature=temp,
    max_tokens=length,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=stop
  )
  return results["choices"][0]["text"].strip(' \n').replace('\n','<BR>')


def executeOpenAIv2(prompt, temp, length, stop=[]):
  if len(stop)==0:
    stop=None
  results = openai.Completion.create(
    engine=P_ENGINE,
    prompt=prompt,
    temperature=temp,
    max_tokens=length,
    top_p=1,
    frequency_penalty=0.4,
    presence_penalty=0.7,
    stop=stop
  )
  return results["choices"][0]["text"].strip(' \n').replace('\n','<BR>')


def removeBadKeywords(text):
  badwords = ['Accessories','Scenery','Clothing', 'Apparel','Person','Human','Furniture','Apparel','Child','Kid','Female','Man']
  allwords = text.split(',')
  newwords = []
  for text_word in allwords:
      text_word=text_word.replace('"','').strip()
    
      if text_word not in badwords:
        newwords.append(text_word)
  return ', '.join(newwords)

def loadConfiguration(filename):
  settings={}
  config={}
  try:
      #with open(filename, "r",encoding='utf-8') as fh:
      #  text = fh.read()
      #  config = json.loads(text)
      with open("settings.json", "r",encoding='utf-8') as fh:
        text = fh.read()
        settings = json.loads(text)
        openai.api_key = settings['openaiApiKey']
  except:
    print ("Failure, could not load settings.json or config.json")
  return settings, config

def saveOutput(html,json=None):
  global FILENAME
  file_object = open(FILENAME+'.html', 'a', encoding="utf-8")
  file_object.write(html)
  file_object.close()
  if json is not None:
    saveJSON(html,json)

def saveJSON(html,jsondata):
  global FILECOUNTER
  filename = jsondata['filename']
  file_object = open('json\\'+filename+'_'+str(FILECOUNTER)+'.json', 'w', encoding="utf-8")
  file_object.write(json.dumps(jsondata))
  file_object.close()
  file_object = open('html\\'+filename+'_'+str(FILECOUNTER)+'.html', 'w', encoding="utf-8")
  file_object.write(html)
  file_object.close()
  FILECOUNTER+=1

def createTitle(product, keyword, boat, version, sentence):
  sentence = sentence.replace('[PRODUCT]',product)
  sentence = sentence.replace('[KEYWORD]',keyword)
  sentence = sentence.replace('[BOAT]',boat)
  sentence = sentence.replace('[VERSION]',version)
  return sentence

def createText(product, keyword, boat, version, sentence, temp, length):
  sentence = sentence.replace('[PRODUCT]',product)
  sentence = sentence.replace('[KEYWORD]',keyword)
  sentence = sentence.replace('[BOAT]',boat)
  sentence = sentence.replace('[VERSION]',version)
  line = executeOpenAI( sentence,
      temp,
      length
    )
  return line

def createTextv2(product, keyword, boat, version, sentence, temp, length):
  sentence = sentence.replace('[PRODUCT]',product)
  sentence = sentence.replace('[KEYWORD]',keyword)
  sentence = sentence.replace('[BOAT]',boat)
  sentence = sentence.replace('[VERSION]',version)
  line = executeOpenAIv2( sentence,
      temp,
      length
    )
  return line

def createInternalSupportDocument(product, keyword,boat, version,title_sentence, problem_sentence, symptoms_sentence,instructions_sentence, temp=0.7, length=500):
  HTML=''
  #format:
  # TITLE
  # PROBLEM
  # SYMPTOMS
  # INSTRUCTIONS
  print('createInternalSupportDocument '+product+'/'+boat+'/'+version)
  title=''
  problem=''
  symptoms=''
  instructions=''
  title=createTitle(product, keyword, boat, version,title_sentence)
  problem=createTextv2(product, keyword, boat, version, problem_sentence, temp, length)
  instructions_sentence = instructions_sentence.replace('[PROBLEM]',problem)
  symptoms_sentence = symptoms_sentence.replace('[PROBLEM]',problem)
  symptoms=createTextv2(product, keyword,boat, version, symptoms_sentence, temp, length)
  instructions=createTextv2(product, keyword,boat, version, instructions_sentence, temp, length)
  HTML = '<HR>'
  HTML+='<h2>createInternalSupportDocument</h2>'
  HTML+='<table>'
  HTML+='<tr><td>Product:</td><td>'+product+'</td></tr>'
  HTML+='<tr><td>Keyword:</td><td>'+keyword+'</td></tr>'
  HTML+='<tr><td>Boat:</td><td>'+boat+'</td></tr>'
  HTML+='<tr><td>Version:</td><td>'+version+'</td></tr>'
  HTML+='<tr><td>Title Sentence:</td><td>'+title_sentence+'</td></tr>'
  HTML+='<tr><td>Problem Sentence:</td><td>'+problem_sentence+'</td></tr>'
  HTML+='<tr><td>Symptoms Sentence:</td><td>'+symptoms_sentence+'</td></tr>'
  HTML+='<tr><td>Instructions Sentence:</td><td>'+instructions_sentence+'</td></tr>'
  HTML+='<tr><td>Temp:</td><td>'+str(temp)+'</td></tr>'
  HTML+='<tr><td>Length:</td><td>'+str(length)+'</td></tr>'
  HTML+='</table><hr>'
  saveOutput(HTML, None)

  HTML='<h1>'+title+'</h1>'
  HTML+='<h4>For model: '+boat+', version: '+version+'</h4>'
  HTML+='<h3>Problem</h3>'
  HTML+='<p class="problem">'+problem+"</p>"
  HTML+='<hr>'
  HTML+='<h3>Symptoms</h3>'
  HTML+='<p class="symptoms">'+symptoms+"</p>"
  HTML+='<hr>'
  HTML+='<h3>How to solve</h3>'
  HTML+='<p class="instructions">'+instructions+"</p>"
  jsond={}
  jsond['filename']='IS_'+boat+'_'+version+'_'+keyword
  jsond['title']=title
  jsond['boat']=boat
  jsond['product']=product
  jsond['version']=version
  jsond['problem']=problem
  jsond['symptoms']=symptoms
  jsond['instructions']=instructions
  saveOutput(HTML, jsond)
  return HTML

def createQADocument(product, keyword, boat, version,title_sentence, sentence, temp=0.7, length=500):
  print('createQADocument '+product+'/'+boat+'/'+version)
  title=''
  answer=''
  title=createTitle(product, keyword,boat, version, title_sentence)
  answer=createTextv2(product, keyword,boat, version, sentence, temp, length)
  HTML = '<HR>'
  HTML+='<h2>createQADocument</h2>'
  HTML+='<table>'
  HTML+='<tr><td>Product:</td><td>'+product+'</td></tr>'
  HTML+='<tr><td>Keyword:</td><td>'+keyword+'</td></tr>'
  HTML+='<tr><td>Boat:</td><td>'+boat+'</td></tr>'
  HTML+='<tr><td>Version:</td><td>'+version+'</td></tr>'
  HTML+='<tr><td>Title Sentence:</td><td>'+title_sentence+'</td></tr>'
  HTML+='<tr><td>Answer Sentence:</td><td>'+sentence+'</td></tr>'
  HTML+='<tr><td>Temp:</td><td>'+str(temp)+'</td></tr>'
  HTML+='<tr><td>Length:</td><td>'+str(length)+'</td></tr>'
  HTML+='</table><hr>'
  saveOutput(HTML)
  
  HTML='<h1>'+title+'</h1>'
  HTML+='<h4>For model: '+boat+', version: '+version+'</h4>'
  HTML+='<h3>Answer</h3>'
  HTML+='<p class="answer">'+answer+"</p>"
  HTML+='<hr>'

  jsond={}
  jsond['filename']='QA_'+boat+'_'+version+'_'+keyword
  jsond['product']=product
  jsond['title']=title
  jsond['boat']=boat
  jsond['version']=version
  jsond['answer']=answer
  saveOutput(HTML, jsond)
  return HTML

def createQADocuments(product, boat, version,temp=0.7, length=500):
  keywords=['shut down','reset','wipe','update','upgrade','install']
  sentences=[
  'Create instructions on how to [KEYWORD] the [PRODUCT].',
  'How do you [KEYWORD] the [PRODUCT]?']
  for sentence in sentences:
    for keyword in keywords:
      doc = createQADocument(product, keyword, boat, version,sentence, sentence, temp, length)

def createSpecificQADocuments(product, keywords, sentences, boat, version,temp=0.7, length=500):
  for sentence in sentences:
    for keyword in keywords:
      doc = createQADocument(product, keyword, boat, version,sentence, sentence, temp, length)

def createCaseComment(product, boat,version,symptoms_sentence,refcase, account, name, temp, length):
  print('createCaseComment '+product+'/'+boat+'/'+version)
  HTML=''
  HTML+='<h2>createCaseComment</h2>'
  HTML+='<table>'
  HTML+='<tr><td>Product:</td><td>'+product+'</td></tr>'
  HTML+='<tr><td>Boat:</td><td>'+boat+'</td></tr>'
  HTML+='<tr><td>Version:</td><td>'+version+'</td></tr>'
  HTML+='<tr><td>Temp:</td><td>'+str(temp)+'</td></tr>'
  HTML+='<tr><td>Length:</td><td>'+str(length)+'</td></tr>'
  HTML+='</table><hr>'
  saveOutput(HTML)
  HTML=''
  sentence = 'Create a response for the problem: '+symptoms_sentence
  comment=createTextv2(product, '',boat, version, sentence, temp, length)
  # HTML='<h1>Comment on case by Support</h1>'
  # HTML+='<h4>For model: '+boat+'</h4>'
  # HTML+='<p class="version">Installed version: '+version+"</p>"
  # HTML+='<h3>Comment</h3>'
  # HTML+='<p class="comment">'+comment+"</p>"

  jsond={}
  jsond['filename']='CASECOMM_'+boat+'_'+version+'_'+refcase
  jsond['comment']=comment
  jsond['case']=refcase
  saveOutput(HTML, jsond)
  return jsond

def createCase(product, keyword,boat,version, title_sentence, problem_sentence, symptoms_sentence, alltitles, temp=0.7, length=500):
  global FILECOUNTER
  global ACCOUNT_POINTER
  global ALL_ACCOUNTS
  print('createCase '+product+'/'+boat+'/'+version)
  HTML=''
  #format:
  # TITLE
  # PROBLEM
  # SYMPTOMS
  print(ALL_ACCOUNTS[ACCOUNT_POINTER])
  account=ALL_ACCOUNTS[ACCOUNT_POINTER]['Account']
  name=ALL_ACCOUNTS[ACCOUNT_POINTER]['Contact']
  print ("For Account/Contact: "+account+'/'+name)
  ACCOUNT_POINTER += 1
  if ACCOUNT_POINTER>len(ALL_ACCOUNTS):
    ACCOUNT_POINTER = 0
  title=''
  problem=''
  symptoms=''
  title_sentence = title_sentence.replace('[ACCOUNT]',account)
  title_sentence = title_sentence.replace('[NAME]',name)
  title=createTitle(product, keyword,boat, version, title_sentence)

  #title=createText(product, keyword,boat, version, title_sentence, temp, length)
  counter=0
  # while title in alltitles:
  #   print("Already got this title, generating a new one")
  #   title=createText(product, keyword,boat, version, title_sentence, temp,length)
  #   counter+=1
  #   if counter>10:
  #     break
  alltitles.append(title)
  #problem_sentence = problem_sentence.replace('[TITLE]',title)
  #problem=createText(product, keyword,boat, version, problem_sentence, temp, length)
  problem = title
  symptoms_sentence = symptoms_sentence.replace('[PROBLEM]',problem)
  symptoms_sentence = symptoms_sentence.replace('[ACCOUNT]',account)
  symptoms_sentence = symptoms_sentence.replace('[NAME]',name)
  symptoms=createTextv2(product, keyword,boat, version, symptoms_sentence, temp, length)
  HTML = '<HR>'
  HTML+='<h2>createCase</h2>'
  HTML+='<table>'
  HTML+='<tr><td>Product:</td><td>'+product+'</td></tr>'
  HTML+='<tr><td>Keyword:</td><td>'+keyword+'</td></tr>'
  HTML+='<tr><td>Boat:</td><td>'+boat+'</td></tr>'
  HTML+='<tr><td>Version:</td><td>'+version+'</td></tr>'
  HTML+='<tr><td>Title Sentence:</td><td>'+title_sentence+'</td></tr>'
  HTML+='<tr><td>Problem Sentence:</td><td>'+problem_sentence+'</td></tr>'
  HTML+='<tr><td>Symptoms Sentence:</td><td>'+symptoms_sentence+'</td></tr>'
  HTML+='<tr><td>Temp:</td><td>'+str(temp)+'</td></tr>'
  HTML+='<tr><td>Length:</td><td>'+str(length)+'</td></tr>'
  HTML+='</table><hr>'
  saveOutput(HTML)

  HTML='<h1>'+title+'</h1>'
  HTML+='<h4>For model: '+boat+'</h4>'
  HTML+='<p class="version">Installed version: '+version+"</p>"
  HTML+='<h3>Problem</h3>'
  HTML+='<p class="problem">'+problem+"</p>"
  HTML+='<hr>'
  HTML+='<h3>Symptoms</h3>'
  HTML+='<p class="symptoms">'+symptoms+"</p>"

  jsond={}
  jsond['filename']='CASE_'+boat+'_'+version+'_'+keyword
  jsond['case']='C1'+str(FILECOUNTER).zfill(6)
  jsond['title']=title
  jsond['boat']=boat
  jsond['product']=product
  jsond['account']=account
  jsond['name']=name
  jsond['version']=version
  jsond['problem']=problem
  jsond['symptoms']=symptoms
  jsond['comments']=createCaseComment(product, boat,version,symptoms,jsond['case'],account,name, temp, length)
  HTML+='<h3>Comment</h3>'
  HTML+='<p class="comment">'+jsond['comments']['comment']+"</p>"
  saveOutput(HTML, jsond)

  return alltitles

def createWebsite(descr, listoffeatures, pdf, product, boat, relatedparts,version, temp=0.7, length=500):
  #Create a Page in the website. Should contain:
  #https://www.westmarine.com/buy/garmin--1042xsv-multifunction-display-with-gt52-hw-transducer--19555341?recordNum=7

  # Product information for selling
  # Product Q&A
  HTML = '<HR>'
  HTML+='<h2>createWebsite</h2>'
  HTML+='<table>'
  HTML+='<tr><td>Product:</td><td>'+product+'</td></tr>'
  HTML+='<tr><td>Boat:</td><td>'+boat+'</td></tr>'
  HTML+='<tr><td>Version:</td><td>'+version+'</td></tr>'
  HTML+='<tr><td>Temp:</td><td>'+str(temp)+'</td></tr>'
  HTML+='<tr><td>Length:</td><td>'+str(length)+'</td></tr>'
  HTML+='</table><hr>'
  saveOutput(HTML)
  HTML=''
  #format:
  # <h1>TITLE</h1>
  # <p>Boat, Version</p>
  # <img>
  # Description
  # Specs
  # Product Overview (listoffeatures, then explained features)
  # Related documents - Installation PDF
  HTML='<h1>'+''+product+' (Version: '+version+'), for model: '+boat+'.</h1>'
  HTML+='<hr>'
  HTML+= '<h3>Specific for model: '+boat+'</h3>'
  HTML+= '<h3>Software version: '+version+'</h3>'
  HTML+= '<h2>Description</h2>'
  HTML+='<p class="content">'+descr+"</p>"
  HTML+='<hr>'
  HTML+= '<h2>Features.</h2>'
  for feature in listoffeatures:
    HTML+='<p class="content"><ul><li>'+feature['feature']+"</li></ul></p>"
  HTML+='<hr>'
  for feature in listoffeatures:
        HTML+= '<h2>'+feature['feature']+'</h2>'
        HTML+='<p class="content">'+feature['explained']+"</p>"
        HTML+='<hr>'
  HTML+= '<h2>Related Documents.</h2>'
  HTML+='<p class="content">'+pdf+"</p>"
  HTML+='<hr>'
  HTML+= '<h2>Related parts.</h2>'
  HTML+='<p class="content">'+'<BR>'.join(relatedparts)+"</p>"
  jsond={}
  jsond['filename']='WEB_'+boat+'_'+version
  jsond['title']=''+product+' (Version: '+version+'), for model: '+boat
  jsond['boat']=boat
  jsond['version']=version
  jsond['product']=product
  jsond['descr']=descr
  jsond['allfeatures']=listoffeatures
  jsond['parts']=relatedparts
  
  jsond['pdf']=pdf
  saveOutput(HTML, jsond)
 
  return HTML

def createPDFDocument(product, keyword, boat, menus, screens, buttons,relatedparts, version, temp=0.7, length=500):
  #Create a PDF document for the installation manuals
  # 
  #https://xhtml2pdf.readthedocs.io/en/latest/usage.html#using-with-python-standalone
  print('createPDFDocument '+product+'/'+boat+'/'+version)
  HTML = '<HR>'
  HTML+='<h2>createPDFDocument</h2>'
  HTML+='<table>'
  HTML+='<tr><td>Product:</td><td>'+product+'</td></tr>'
  HTML+='<tr><td>Keyword:</td><td>'+keyword+'</td></tr>'
  HTML+='<tr><td>Boat:</td><td>'+boat+'</td></tr>'
  HTML+='<tr><td>Version:</td><td>'+version+'</td></tr>'
  HTML+='<tr><td>Temp:</td><td>'+str(temp)+'</td></tr>'
  HTML+='<tr><td>Length:</td><td>'+str(length)+'</td></tr>'
  HTML+='</table><hr>'
  saveOutput(HTML)

  HTML=''
  filename=''
  # sentence = 'Create a list of software buttons for [PRODUCT].'
  # controls=createTextv2(product, keyword, boat, version,sentence, 0.5, length)
  # sentence = 'Create a list with names of all the screens for: [PRODUCT].'
  # screens=createTextv2(product, keyword, boat, version,sentence, 0.5, length)
  # sentence = "Create a list with names of all the menu's for: [PRODUCT]."
  # menus=createTextv2(product, keyword,boat, version, sentence, 0.5, length)
  # # sentence = 'Create instructions on how to connect the transducer to [PRODUCT] device.'
  # transducer=createText(product, keyword,boat, version, sentence, temp, length)
  # sentence = 'Create instructions on how to wire the power connection to [PRODUCT] device.'
  # power=createText(product, keyword,boat, version, sentence, temp, length)
  # sentence = 'Create instructions on how to connect a device to the serial connection port of [PRODUCT] device.'
  # serial=createText(product, keyword,boat, version, sentence, temp, length)
  # sentence = 'Create instructions on how to configure the software for [PRODUCT] device.'
  # software=createText(product, keyword,boat, version, sentence, temp, length)
  sentence = 'Explain in detail how to setup the wifi for [PRODUCT].'
  wifi=createTextv2(product, keyword,boat, version, sentence, temp, 2000)
  # sentence = 'Create instructions on how to configure the radar for [PRODUCT] device.'
  # radar=createText(product, keyword, boat, version,sentence, temp, length)
  sentence = 'Create troubleshoot instructions for [PRODUCT].'
  trouble=createTextv2(product, keyword, boat, version,sentence, temp, length)
  sentence = 'Create release notes for [PRODUCT] with version [VERSION].'
  releasenotes=createTextv2(product, keyword, boat, version,sentence, temp, length)

  HTML='<h1>'+'Installation manual for: '+product+' (Version: '+version+'), model: '+boat+'.</h1>'
  HTML+='<hr>'
  HTML+= '<h2>Main Software Buttons.</h2>'
  HTML+='<p class="content">'+'<BR>'.join(buttons)+"</p>"
  HTML+='<hr>'
  HTML+= '<h2>List of available Software screens.</h2><p>'
  for screen in screens:
    HTML+=''+screen+"<BR>"
  HTML+='</p><hr>'
  allscreenexplanations=[]
  for screen in screens:
    try:
        HTML+= '<h3>Screen: '+screen+'</h3>'
        sentence = 'Explain the function of the screen: '+screen+' with the [PRODUCT].'
        instruct=createTextv2(product, keyword, boat, version,sentence, 0.5, length)
        allscreenexplanations.append({"screen":screen, "instruction":instruct})
        HTML+='<p class="content">'+instruct+"</p>"
        HTML+='<hr>'
    except:
      print("Error with: "+screen)
      pass
  
  # allscreens = screens.split('<BR>')
  # allscreenexplanations=[]
  # regex = r"^([\d.)\- ]+)"
  # for screen in allscreens:
  #   try:
  #     screen = re.sub(regex, '-', screen, 0, re.MULTILINE).strip()
  #     thescreen = screen.replace('.','')
  #     if not thescreen=='' and thescreen.startswith('-'):
  #       thescreen = thescreen.replace('-',' ').strip().title()
  #       HTML+= '<h3>Screen: '+thescreen+'</h3>'
  #       sentence = 'Explain the function of the screen: '+thescreen+' with the [PRODUCT].'
  #       instruct=createTextv2(product, keyword, boat, version,sentence, 0.5, length)
  #       allscreenexplanations.append({"screen":thescreen, "instruction":instruct})
  #       HTML+='<p class="content">'+instruct+"</p>"
  #       HTML+='<hr>'
  #   except:
  #     print("Error with: "+screen)
  #     pass
  HTML+= '<h2>List of available Menus.</h2><p>'
  for menu in menus:
    HTML+=''+menu["main"]+"<BR>"
  HTML+='</p><hr>'
  allmenuexplanations=[]
  for menu in menus:
    try:
        HTML+= '<h3>Menu: '+menu["main"]+'</h3>'
        sentence = 'Explain to me in detail how to use: the '+menu["main"]+' menu with the [PRODUCT].'
        instruct=createTextv2(product, keyword, boat, version,sentence, 0.5, length)
        allmenuexplanations.append({"menu":menu, "instruction":instruct})
        HTML+='<p class="content">'+instruct+"</p>"
        submenus=menu["sub"].split(';')
        for submenu in submenus:
          HTML+= '<h4>Sub Menu: '+submenu+'</h4>'
          sentence = 'Explain to me in detail how to use: the '+menu["main"]+', '+submenu+' menu with the [PRODUCT].'
          instruct=createTextv2(product, keyword, boat, version,sentence, 0.5, length)
          allmenuexplanations.append({"menu":menu["main"]+'-'+submenu, "instruction":instruct})
          HTML+='<p class="content">'+instruct+"</p>"
          HTML+='<hr>'
    except:
      print("Error with: "+menu)
      pass
  # menus = menus.replace(product,'')
  # allmenus = menus.split('<BR>')
  # allsmenuexplanations=[]
  # for menu in allmenus:
  #   try:
  #     menu = re.sub(regex, '-', menu, 0, re.MULTILINE).strip()
  #     themenu = menu.replace('.','')
  #     if not themenu==''  and themenu.startswith('-'):
  #       themenu = themenu.replace('-',' ').strip().title()
  #       HTML+= '<h3>Menu: '+themenu+'</h3>'
  #       sentence = 'Explain the menu: '+themenu+' with the [PRODUCT].'
  #       instruct=createTextv2(product, keyword, boat, version,sentence, 0.5, length)
  #       allsmenuexplanations.append({"menu":themenu, "instruction":instruct})
  #       HTML+='<p class="content">'+instruct+"</p>"
  #       HTML+='<hr>'
  #   except:
  #     print("Error with: "+menu)
  #     pass

  # HTML+= '<h2>How to connect the transducer.</h2>'
  # HTML+='<p class="content">'+transducer+"</p>"
  # HTML+='<hr>'
  # HTML+= '<h2>How to wire the power connection.</h2>'
  # HTML+='<p class="content">'+power+"</p>"
  # HTML+='<hr>'
  # HTML+= '<h2>How to connect a device to the serial port.</h2>'
  # HTML+='<p class="content">'+serial+"</p>"
  # HTML+='<hr>'
  # HTML+= '<h2>Install and configure the software.</h2>'
  # HTML+='<p class="content">'+software+"</p>"
  # HTML+='<hr>'
  HTML+= '<h2>How to setup the Wifi on the device.</h2>'
  HTML+='<p class="content">'+wifi+"</p>"
  # HTML+='<hr>'
  # HTML+= '<h2>How to setup the radar.</h2>'
  # HTML+='<p class="content">'+radar+"</p>"
  HTML+='<hr>'
  HTML+= '<h2>Troubleshooting.</h2>'
  HTML+='<p class="content">'+trouble+"</p>"
  HTML+='<hr>'
  HTML+= '<h2>Release notes for version '+version+'.</h2>'
  HTML+='<p class="content">'+releasenotes+"</p>"
  HTML+='<hr>'
  HTML+= '<h2>Related parts.</h2>'
  HTML+='<p class="content">'+'<BR>'.join(relatedparts)+"</p>"

  jsond={}
  jsond['filename']='PDF_'+boat+'_'+version+'_'+keyword
  jsond['title']='Installation manual for: '+product+' (Version: '+version+'), model: '+boat
  jsond['boat']=boat
  jsond['product']=product
  jsond['version']=version
  jsond['buttons']=buttons
  jsond['screens']=allscreenexplanations
  jsond['menus']=allmenuexplanations
  jsond['wifi']=wifi
  jsond['trouble']=trouble
  jsond['release']=releasenotes
  jsond['parts']=relatedparts
  saveOutput(HTML, jsond)
  filename = jsond['filename']

  return filename

def createWordDocument(product, keyword, boat,menus, screens, buttons,features,relatedparts, version,temp=0.7, length=500):
  #Create a Word Document with Product Information sheet. 
  # Description
  # Features
  # Explain each feature
  #https://python-docx.readthedocs.io/en/latest/
  print('createWordDocument '+product+'/'+boat+'/'+version)
  filename=''
  HTML = '<HR>'
  HTML+='<h2>createWordDocument</h2>'
  HTML+='<table>'
  HTML+='<tr><td>Product:</td><td>'+product+'</td></tr>'
  HTML+='<tr><td>Keyword:</td><td>'+keyword+'</td></tr>'
  HTML+='<tr><td>Boat:</td><td>'+boat+'</td></tr>'
  HTML+='<tr><td>Version:</td><td>'+version+'</td></tr>'
  HTML+='<tr><td>Temp:</td><td>'+str(temp)+'</td></tr>'
  HTML+='<tr><td>Length:</td><td>'+str(length)+'</td></tr>'
  HTML+='</table><hr>'
  saveOutput(HTML)

  sentence = 'Describe the product [PRODUCT].'
  descr=createTextv2(product, keyword, boat, version,sentence, 0.5, length)
  sentence = 'Create a list of the top software functions for the [PRODUCT].'
  #overview=createTextv2(product, keyword,boat, version, sentence, temp, 700)
  overview = ''
  overview = '<BR>'.join(features)
  sentence = 'Create a list of functionality for the [PRODUCT].'
  #features=createTextv2(product, keyword,boat, version, sentence, temp, 700)
  #features = []
  #for menu in menus:
  #  features.append(menu["main"])


  HTML='<h1>'+'Product Information sheet for: '+product+' (Version: '+version+'), model: '+boat+'.</h1>'
  HTML+='<hr>'
  HTML+= '<h2>Description.</h2>'
  HTML+='<p class="content">'+descr+"</p>"
  HTML+='<hr>'
  HTML+= '<h2>Overview.</h2>'
  HTML+='<p class="content">'+overview+"</p>"
  HTML+='<hr>'
  HTML+= '<h2>Highlights.</h2>'
  #features = features.replace(product,'')
  #HTML+='<p class="content">'+'<BR>'.join(features)+"</p>"
  #HTML+='<hr>'
  #allfeatures=features.split('<BR>')
  listoffeatures=[]
  #regex = r"^([\d.)\- ]+)"

  for feature in features:
    #feature = re.sub(regex, '-', feature, 0, re.MULTILINE).strip()
    feature = feature.replace('.','')
    #if not feature==''  and feature.startswith('-'):
    feature = feature.replace('-',' ').strip().title()
    HTML+= '<h3>'+feature+'</h3>'
    sentence = 'Explain to me the feature: '+feature+' in the [PRODUCT].'
    explained=createTextv2(product, keyword,boat, version, sentence, 0.9, length)
    explained = explained.replace(feature,'')
    listoffeatures.append({"feature":feature, "explained":explained})
    HTML+='<p class="content">'+explained+"</p>"
    HTML+='<hr>'

  HTML+='<hr>'
  HTML+= '<h2>Related parts.</h2>'
  HTML+='<p class="content">'+'<BR>'.join(relatedparts)+"</p>"

  jsond={}
  jsond['filename']='WORD_'+boat+'_'+version+'_'+keyword
  jsond['title']='Product Information sheet for: '+product+' (Version: '+version+'), model: '+boat
  jsond['boat']=boat
  jsond['version']=version
  jsond['descr']=descr
  jsond['overview']=overview
  jsond['product']=product
  jsond['features']=listoffeatures
  jsond['parts']=relatedparts
 
  saveOutput(HTML, jsond)


  return descr, listoffeatures


def createGuidareDocuments(boat, version,temp=0.8, length=500):
  global FILECOUNTER
  global FILENAME
  FILECOUNTER=6000
  FILENAME='Report-Guidare'
  alltitles=[]
  product = "Guidare Boat Navigation Software Suite"
  relatedparts=["Operator Panel","GPS","Radar","Sonar","Panoptix Sonar","Sensors"]
  menus=[{"main":"Homescreen","sub":"User Profiles;Settings;Man Overboard;Alarms;GPS Settings;Status Area;Sidebar"},
  {"main":"Autopilot control","sub":"Locked heading;Navigation;Disengaging the autopilot"},
  {"main":"Chart app","sub":"Modes;Vessel details;View and motion;Placing a waypoint;Creating a route;Autorouting;Create a track"},
  {"main":"Weather mode","sub":"Modes;Animated weather"},
  {"main":"Sonar app","sub":"Modes;3D Controls;Sonar Channels"},
  {"main":"Radar app","sub":"Modes;Controls;AIS targets;Guard zone alarms;FishFinder"},
  ]
  screens=["Main","Navigation","Radar Map","Maps and Charts","Tides and Currents","FishFinder","Wind and Weather","Settings"]
  buttons=["Home","Route","Maps","Settings","FishFinder","Quit"]
  features=["Connect to your favorite third-party devices",
            "Navigate any waters with preloaded mapping and coastal charts",
            "Manage your marine experience from nearly anywhere",
            "Connect traditional and scanning sonars",
            "Panoptix Sonar support",
            "BlueChart and Vision Chart support",
            "Built in Wifi",
            "FishFinder"]
  #********************** QA ******************************
  if not CASES_ONLY:
    createQADocuments(product,boat, version, temp, length)
    
    keywords=['enable night mode with','disable night mode with','disable real time tracking with','enable real time tracking with','disable the fishfinder function with','enable the doppler radar with']
    sentences=['How do you [KEYWORD] the [PRODUCT]?','Create instructions on how to [KEYWORD] the [PRODUCT].']
    createSpecificQADocuments(product,keywords, sentences,boat, version, temp, length)
    
    keywords=['real time tracking','wind capturing','monitoring the bottom of the ocean','identification of vessels']
    sentences=['How does [KEYWORD] work with the [PRODUCT]?']
    createSpecificQADocuments(product,keywords, sentences,boat, version, temp, length)

    keywords=['real time tracking','wind capturing','monitoring the bottom of the ocean','identification of vessels']
    sentences=['Explain to me how [KEYWORD] works with the [PRODUCT]?']
    createSpecificQADocuments(product,keywords, sentences, boat, version,temp, length)

    #********************** Internal Support Documents *********
    title_sentence='How to solve [KEYWORD] for [PRODUCT] on a [BOAT].'
    problem_sentence='Create a description for this problem: The [PRODUCT] has [KEYWORD]. '
    symptoms_sentence='What are the symptoms for the following problem: [PROBLEM]'
    instructions_sentence='Create a solution for the following problem: [PROBLEM]'
    keywords=['a stuck device','a blank screen','a blinking red led','a spinning progress bar','a map not being displayed','no connection to sensors']
    for keyword in keywords:
      createInternalSupportDocument(product,keyword, boat, version,title_sentence, problem_sentence,symptoms_sentence, instructions_sentence, 0.8, length)
      # break

  #********************** Cases Documents *********
  #NAME, ACCOUNT
  titlewords=['I need a fix for [KEYWORD] on my [PRODUCT].',
  'When using the product [PRODUCT] on my [BOAT], [KEYWORD].']
  title_sentence='[TITLEWORD]'
  problem_sentence='Create a description for this problem: [TITLE] for the product [PRODUCT].'
  symptoms_sentence='Create a detailed problem description for: [PROBLEM]'
  keywords=['the device is stuck','i have a blank screen','there is a blinking red led','the progress bar keeps spinning',
  'the map is not being displayed','there is no connection to the sensors','cannot connect to Panoptix device','cannot connect other devices','cannot connect to built in wifi']
  for titleword in titlewords:
    for keyword in keywords:
      title = title_sentence
      title = title.replace('[TITLEWORD]',titleword)
      alltitles=createCase(product,keyword, boat, version,title, problem_sentence,symptoms_sentence, alltitles, 1.2, length)
    #   break
    # break

  #********************** Cases Documents *********
  titlewords=[
  'On my [BOAT], how do i fix [KEYWORD] is [OPERATION] the [PRODUCT] application.',
  'How do i fix: [KEYWORD] is [OPERATION] the [PRODUCT] application.']
  title_sentence='[TITLEWORD]'
  problem_sentence='Create a description for this problem: [TITLE] for the product [PRODUCT]. '
  symptoms_sentence='Create a detailed problem description for: [PROBLEM]'
  keywords=['enable the real time tracker','disable the wind capture function','activating the vessel identification'
  ,'changing the maps', 'mapping a new route','connecting the sonar','enabling the autopilot','enabling the fishfinder','connecting to the internal wifi',
  'viewing the BlueChart maps','connecting to sonar']
  operations=['shutting down','freezing','resetting','booting','erroring']
  for operation in operations:
   for titleword in titlewords:
    for keyword in keywords:
      title = title_sentence
      title = title.replace('[TITLEWORD]',titleword)
      title = title.replace('[OPERATION]',operation)
      alltitles=createCase(product,keyword,boat, version, title, problem_sentence,symptoms_sentence, alltitles, 1.2, length)
  #     break
  #   break
  #  break

  if not CASES_ONLY:

    #********************** PDF ******************************
    filenamepdf=createPDFDocument(product, '', boat, menus, screens, buttons,relatedparts, version, temp, length)
    #********************** Word ******************************
    descr, listoffeatures=createWordDocument(product, '', boat, menus, screens, buttons,features,relatedparts, version, temp, length)
    #********************** Website ******************************
    # Create Website version from the WORD documents
    createWebsite(descr, listoffeatures, filenamepdf, product, boat, relatedparts,version, temp, length)


def createAssistentDocuments(boat, version,temp=0.8, length=500):
  global FILECOUNTER
  global FILENAME
  FILECOUNTER=4000
  FILENAME='Report-Assistente'

  alltitles=[]
  product = "Assistente Boat Assisted Software Suite"
  relatedparts=["FLIR cameras","GPS receiver","Operator Panel","Emergency stop button","Heading sensors"]
  menus=[{"main":"Assistent","sub":"Camera setup;GPS setup;Maps;Birdseye configuration"},
  {"main":"Docking","sub":"Views;Birdseye;Help me"},
  {"main":"Tools","sub":"Setup;Connected Cameras;Connected GPS;Upgrade;Reset"},
  {"main":"Emergency","sub":"Reset;Setup"},
  ]
  screens=["Main","Assistent","Docker","Birdseye","Settings"]
  buttons=["Home","Emergency Stop","Assistent","Dock","Settings","Quit"]
  features=["Connect your stereo FLIR cameras and GPS",
            "Detects nearby obstacles both visually and audibly",
            "Uses AI for object detection",
            "Creates a virtual vender around your boat",
            "Connect your propulsion system using NMEA2000 networking protocols",
            "Supports Single, Three or Five camera setups"
]
  #********************** QA ******************************
  if not CASES_ONLY:
    createQADocuments(product,boat, version, temp, length)
    
    keywords=['disable the stern camera with','enable all cameras with','disable the gps with','execute an emergency stop with']
    sentences=['How do you [KEYWORD] the [PRODUCT]?','Create instructions on how to [KEYWORD] the [PRODUCT].']
    createSpecificQADocuments(product,keywords, sentences,boat, version, temp, length)
    
    keywords=['assisted docking','collision detection','distance calculation within the dock','automating the propulsion system']
    sentences=['How does [KEYWORD] work with the [PRODUCT]?']
    createSpecificQADocuments(product,keywords, sentences,boat, version, temp, length)

    keywords=['assisted docking','collision detection','distance calculation within the dock','automating the propulsion system']
    sentences=['Explain to me how [KEYWORD] works with the [PRODUCT]?']
    createSpecificQADocuments(product,keywords, sentences, boat, version,temp, length)

    #********************** Internal Support Documents *********
    title_sentence='How to solve [KEYWORD] for [PRODUCT] on a [BOAT].'
    problem_sentence='Create a description for this problem: The [PRODUCT] has [KEYWORD]. '
    symptoms_sentence='What are the symptoms for the following problem: [PROBLEM]'
    instructions_sentence='Create a solution for the following problem: [PROBLEM]'
    keywords=['a blank camera','when there is no GPS signal recevied','a blinking red alert led','when the distance calculation gives wrong results'
    ,'when propulsion is not activated when docking','when the propulsion speed is to high']
    for keyword in keywords:
      createInternalSupportDocument(product,keyword, boat, version,title_sentence, problem_sentence,symptoms_sentence, instructions_sentence, 0.8, length)
      # break

  #********************** Cases Documents *********
  titlewords=['I need a fix for [KEYWORD] on my [PRODUCT].',
  'When using the product [PRODUCT] on my [BOAT], [KEYWORD].']
  title_sentence='[TITLEWORD]'
  problem_sentence='Create a description for this problem: [TITLE] for the product [PRODUCT].'
  symptoms_sentence='Create a detailed problem description for: [PROBLEM]'
  keywords=['the device is stuck','i have a blank screen','there is a blinking warning led','no camera feed on the display',
  'the birdseye view is not being displayed','there is no connection to the engine','propulsion is to low']
  for titleword in titlewords:
    for keyword in keywords:
      title = title_sentence
      title = title.replace('[TITLEWORD]',titleword)
      alltitles=createCase(product,keyword, boat, version,title, problem_sentence,symptoms_sentence, alltitles, 1.2, length)
    #   break
    # break

  #********************** Cases Documents *********
  titlewords=[
  'On my [BOAT], how do i fix [KEYWORD] is [OPERATION] the [PRODUCT] application.',
  'How do i fix: [KEYWORD] is [OPERATION] the [PRODUCT] application.']
  title_sentence='[TITLEWORD]'
  problem_sentence='Create a description for this problem: [TITLE] for the product [PRODUCT]. '
  symptoms_sentence='Create a detailed problem description for: [PROBLEM]'
  keywords=['enable the birdseye view','disable the assistent','activating the assistent'
  ,'propulsion keeps spinning, but we are docked', 'enabling the three camera setup','disable the gps device']
  operations=['shutting down','freezing','resetting','booting','erroring']
  for operation in operations:
   for titleword in titlewords:
    for keyword in keywords:
      title = title_sentence
      title = title.replace('[TITLEWORD]',titleword)
      title = title.replace('[OPERATION]',operation)
      alltitles=createCase(product,keyword,boat, version, title, problem_sentence,symptoms_sentence, alltitles, 1.2, length)
  #     break
  #   break
  #  break

  if not CASES_ONLY:

    #********************** PDF ******************************
    filenamepdf=createPDFDocument(product, '', boat, menus, screens, buttons, relatedparts,version, temp, length)
    #********************** Word ******************************
    descr, listoffeatures=createWordDocument(product, '', boat, menus, screens, buttons,features,relatedparts, version, temp, length)
    #********************** Website ******************************
    # Create Website version from the WORD documents
    createWebsite(descr, listoffeatures, filenamepdf, product, boat, relatedparts,version, temp, length)


def createVedereDocuments(boat, version,temp=0.8, length=500):
  global FILECOUNTER
  global FILENAME
  FILECOUNTER=2000
  FILENAME='Report-Vedere'

  alltitles=[]
  product = "Vedere Vessel Management Software"
  relatedparts=["Console Panels","Alarm System Panel","Marine PC","Trackball","Trackpad","CCTV Cameras","PLC Controllers"]
  menus=[{"main":"Settings","sub":"The Initial Setup;Configuration;Manage External Modules;Manage Alarms"},
  {"main":"Help","sub":"How to Wipe and Zoom;Viewing the Alarm panels;Setup the system;Configure external modules"},
  {"main":"Propulsion","sub":"View Oil Pressure;View Cooling Water Temperature;View Fuel Consumption;View Gearbox Pressure;View Gearbox Temperature;View Ventilators;Setup Auto Stop"},
  {"main":"Navigator Lights","sub":"Manage Bow Lights;Manage Stern Lights;Manage Port Lights;Manage Starboard Lights"},
  {"main":"Ventilation","sub":"View Engines Vents;View Main Vents;View Cabin Vents"},
  {"main":"Tanks","sub":"View Main Fuel Tank;View Secondary Fuel Tank"},
  {"main":"Conning","sub":"View Wind;View Air Temperature;View Water Temperature;View Position;View Depth;View Depth below Keel;View Rudder Position;View Navigator Lights"},
  {"main":"Fire Fighting","sub":"View Pump Room Fans;View Fuel Trim Pumps;View Firefight Engine;View Foam System"},
  {"main":"Generators","sub":"View Main;View Backup"},
  {"main":"Batteries","sub":"View current Voltage;View usage History"},
  {"main":"CCTV","sub":"Manage Bow Camera;Manage Stern Camera;Manage Port Camera;Manage Starboard Camera"},
  {"main":"Diagnostic","sub":"Perform Self Test;Execute soft reset;Execute hard reset"},
  {"main":"Alarms","sub":"View Emergency Alarms;View all Alarms;Reset all Alarms"}]
  screens=["Settings","Propulsion","Navigator Lights","Ventilation","Tanks","Conning","Fire Fighting","Generators","Batteries","CCTV","Diagnostic","Engines","Alarms"]
  buttons=["Home","Alarms","Panels","Quit"]
  features=["Supports Multiple Platforms (Windows, Linux)",
            "Connect to PLC Control Processors",
            "Communicate using NMEA2000 networking",
            "Create and view multiple Operator Panels",
            "Swipe and zoom to pages or the alarm list",
            "4 layers of additional data can be viewed for each panel",
            "Use additional photos for each connected device to easily identify your asset",
            "Available for Commercial Ships, Mega Yachts, Navy Ships",
            "Integrate with ISV (Internet Ship View) for remote communication"
  ]
  #********************** QA ******************************
  if not CASES_ONLY:
    createQADocuments(product,boat, version, temp, length)
    
    keywords=['add a new device to','add a new alarm setting to','diplay the Conning display with','reset all the alarms with','enable the firefight status display with','view the depth of the vessel with']
    sentences=['How do you [KEYWORD] the [PRODUCT]?','Create instructions on how to [KEYWORD] the [PRODUCT].']
    createSpecificQADocuments(product,keywords, sentences,boat, version, temp, length)
    
    keywords=['monitoring the propulsion devices','fire fighting','monitoring the batteries','monitoring the tanks']
    sentences=['How does [KEYWORD] work with the [PRODUCT]?']
    createSpecificQADocuments(product,keywords, sentences,boat, version, temp, length)

    keywords=['setting up the alarms','the conning display','monitoring the devices','connecting a brand new device','adding new hardware']
    sentences=['Explain to me how [KEYWORD] works with the [PRODUCT]?']
    createSpecificQADocuments(product,keywords, sentences, boat, version,temp, length)

    #********************** Internal Support Documents *********
    title_sentence='How to solve [KEYWORD] for [PRODUCT] on a [BOAT].'
    problem_sentence='Create a description for this problem: The [PRODUCT] has [KEYWORD]. '
    symptoms_sentence='What are the symptoms for the following problem: [PROBLEM]'
    instructions_sentence='Create a solution for the following problem: [PROBLEM]'
    keywords=['a stuck device','how do i reset the software','a blank screen','a blinking alarm','cannot add a new device to the product','reset an alarm for a device','changing the alarm boundaries']
    for keyword in keywords:
      createInternalSupportDocument(product,keyword, boat, version,title_sentence, problem_sentence,symptoms_sentence, instructions_sentence, 0.8, length)
      # break

  #********************** Cases Documents *********
  titlewords=['I need a fix for [KEYWORD] on my [PRODUCT].',
  'When using the product [PRODUCT] on my [BOAT], [KEYWORD].']
  title_sentence='[TITLEWORD]'
  problem_sentence='Create a description for this problem: [TITLE] for the product [PRODUCT].'
  symptoms_sentence='Create a detailed problem description for: [PROBLEM]'
  keywords=['the device is stuck','i have a blank screen','i have a blinking alarm','i cannot add a new device to the product',
  'how can i reset an alarm for a device','how do i change the alarm boundaries',
  'how to reset the software','how to remove an existing device','my conning display is stuck']
  for titleword in titlewords:
    for keyword in keywords:
      title = title_sentence
      title = title.replace('[TITLEWORD]',titleword)
      alltitles=createCase(product,keyword, boat, version,title, problem_sentence,symptoms_sentence, alltitles, 1.2, length)
    #   break
    # break

  #********************** Cases Documents *********
  titlewords=[
  'On my [BOAT], how do i fix [KEYWORD] is [OPERATION] the [PRODUCT] application.',
  'How do i fix: [KEYWORD] is [OPERATION] the [PRODUCT] application.']
  title_sentence='[TITLEWORD]'
  problem_sentence='Create a description for this problem: [TITLE] for the product [PRODUCT]. '
  symptoms_sentence='Create a detailed problem description for: [PROBLEM]'
  keywords=['enable the conning display','disable the CCTV on the stern','enable the alarm','adding a new device','reset the alarms',
  'zooming to a different screen','disable the engines','configure the software','removing an old device','installing a brand new device',
  'enable the fire fighting mode']
  operations=['shutting down','freezing','resetting','booting','erroring']
  for operation in operations:
   for titleword in titlewords:
    for keyword in keywords:
      title = title_sentence
      title = title.replace('[TITLEWORD]',titleword)
      title = title.replace('[OPERATION]',operation)
      alltitles=createCase(product,keyword,boat, version, title, problem_sentence,symptoms_sentence, alltitles, 1.2, length)
  #     break
  #   break
  #  break

  if not CASES_ONLY:
    #********************** PDF ******************************
    filenamepdf=createPDFDocument(product, '', boat, menus, screens, buttons,relatedparts, version, temp, length)
    #********************** Word ******************************
    descr, listoffeatures=createWordDocument(product, '', boat, menus, screens, buttons,features,relatedparts, version, temp, length)
    #********************** Website ******************************
    # Create Website version from the WORD documents
    createWebsite(descr, listoffeatures, filenamepdf, product, boat, relatedparts,version, temp, length)


def process(filename):
  #Get All files
  settings, config = loadConfiguration(filename)
  readCSV()
  total=0
  temp = 0.8
  length = 500
  if CASES_ONLY:
    print("REMARK: Only cases will be created")
  #versions=['1','1 beta','2','2 beta','3 beta','3']  
  #boats=['Mercury','Yamaha','Honda','Evinrude','Suzuki','Johnson','Tohatsu','OMC','Chrysler','Force','Mariner','Mercruiser','Mercury','Nissan','Sears']
  versions=['1']  
  boats=['Mercury']
  for boat in boats:
    for version in versions:
      createGuidareDocuments(boat,version, temp, length)
      createVedereDocuments(boat,version, temp, length)
      createAssistentDocuments(boat,version, temp, length)

  print("We are done!\n")
  print("Processed: "+str(total)+" results\n")
  
try:
  #fileconfig = sys.argv[1]
  #process(fileconfig)
  process('')
except Exception as e: 
  print(e)
  traceback.print_exception(*sys.exc_info())
  #print ("Specify configuration json (like config.json) on startup")
