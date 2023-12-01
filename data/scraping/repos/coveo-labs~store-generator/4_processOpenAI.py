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

import glob

P_ENGINE = 'text-curie-001'



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
  return results["choices"][0]["text"].strip(' \n')


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
      with open(filename, "r",encoding='utf-8') as fh:
        text = fh.read()
        config = json.loads(text)
      with open("settings.json", "r",encoding='utf-8') as fh:
        text = fh.read()
        settings = json.loads(text)
  except:
    print ("Failure, could not load settings.json or config.json")
  return settings, config


def createProductName(keywords, brand, gender, color, temp=0.9, length=100):
  printTime("Starting Product Name for:")
  print("Keywords: "+keywords)
  print("Brand   : "+brand)
  print("Gender  : "+gender)
  print("Temp    : "+str(temp))
  print("Length  : "+str(length))
  name=''
  if not gender=='':
    keywords = keywords+', for '+gender
  name = executeOpenAI("""Generate a product name for sales out of keywords.

Keywords: blouse, Sunglasses, Coat, Nature, Woman, Cape, Fashion, Outdoors, Jacket, Cloak, for Woman
Brand: Zara
Color: black
Name: Fashionable black blouse. Great for outdoor adventures, especially for women. Designed by Zara.

Keywords: trouser, Sitting, Door, Porch, Wood, Bench, for Men
Brand: MFH
Color: red
Name: Red modern trouser made for men. Created by MFH.

Keywords: glove, winter
Brand: Puma
Color: red
Name: Well designed red winter gloves, designed by Puma.

Keywords: dress, summer
Brand: Calvin Klein
Color: red
Name: Fashionable red dress for summer. Designed by Calvin Klein.

Keywords: boots
Brand: Mammut
Color: 
Name: By Mammut, Incredible hiking boots.

Keywords: bag
Brand: Addidas
Color: blue
Name: Excellent durable, blue bag made by Addidas.

Keywords: """+keywords+"""
Brand: """+brand+"""
Color: """+color+"""
Name:""",
    temp,
    length, ["\n"]
  )
  print("GENERATED: "+name)
  return name

def createProductDescription(name, temp=0.3, length=200):
  printTime("Starting Product Description for:")
  print("Name    : "+name)
  print("Temp    : "+str(temp))
  print("Length  : "+str(length))
  descr=''
  descr = executeOpenAI("""Write a description for a product based off its name:

Name: """+name,
    temp,
    length
  )
  print("GENERATED: "+descr)
  return descr

def createProductSelling(descr, temp=0.45, length=200):
  printTime("Starting Product Selling for:")
  print("Descr   : "+descr)
  print("Temp    : "+str(temp))
  print("Length  : "+str(length))
  sell=''
  sellingpoints = executeOpenAI("Write three selling points based off a product description:\n\nProduct: An eco-friendly toothbrush made out of bamboo. Comes in several colours and has soft bristles.\nSelling points:\n1. ECO-FRIENDLY - Made from bamboo and can be sustainably disposed due to its biodegradable material.\n2. RESILIANT - The bamboo is treated such to make it significantly durable.\n3. COLORFUL - Comes in a variety of bright colors.\n\nProduct: A desktop lamp in the shape of a dragon.\nSelling points:\n1. FUN - An exciting way to light up your desk.\n2. UNIQUE - A unique design that will stand out from the rest.\n3. BRIGHT - It is built in such a way to ensure it can light any room its placed in.\n\nProduct: "+descr+".\nSelling points:\n1.",
    temp,
    length
  )
  print("GENERATED: "+sellingpoints)
  if "\n2." in sellingpoints and "\n3." in sellingpoints:
    sell = "\t\t<h3>"+sellingpoints.split("\n2.")[0]+"</h3>\n"
    sell += "\t\t<h3>"+sellingpoints.split("\n2.")[1].split("\n3.")[0]+"</h3>\n"
    sell += "\t\t<h3>"+sellingpoints.split("\n2.")[1].split("\n3.")[1]+"</h3>\n"
  return sell

def createProductFeatures(name, temp=1.0, length=200):
  printTime("Starting Product Features for:")
  print("Name    : "+name)
  print("Temp    : "+str(temp))
  print("Length  : "+str(length))
  features = executeOpenAI("List the top features of a "+name+".",
    temp,
    length
  )
  print("GENERATED: "+features)
  return features


def createProductArticle(descr, name, sentence, words, category, temp=0.5, length=500):
  printTime("Starting Product Article for:")
  print("Descr   : "+descr)
  print("Name    : "+name)
  print("Sentence: "+sentence)
  print("Temp    : "+str(temp))
  print("Length  : "+str(length))
  sentences=[]
  #Check if [WORD] is in the sentence
  if '[WORD]' in sentence:
   for word in words:
    line = sentence
    line = line.replace('[WORD]',word)
    line = line.replace('[DESCR]',descr)
    line = line.replace('[CAT]',category)
    line = line.replace('[NAME]',name)
    print(line)
    res = executeOpenAI( line,
      temp,
      length
    )
    if (not res==""):
      sentences.append('<h1>'+line+"</h1>")
      sentences.append('<p>'+res+"</p>")
  else:
    line = sentence
    line = line.replace('[CAT]',category)
    line = line.replace('[NAME]',name)
    line = line.replace('[DESCR]',descr)
    print(line)
    res = executeOpenAI( line,
      temp,
      length
    )
    if (not res==""):
      sentences.append('<h1>'+line+"</h1>")
      sentences.append('<p>'+res+"</p>")
  return sentences


def basicColors():
  mainColors = ["black","silver","gray","white","maroon","red","purple","fuchsia","green","lime","olive","yellow","navy","blue","teal","aqua"]
  return mainColors

def normalizeColors(color):
   mainColors = basicColors() 
   for col in mainColors:
     if col in color:
       color += ' '+col
   return color.title()

def detect_openai(data, settings, config):
    #create random brand
    if ('cat_retailer' not in data):
      if 'rand_retailer' in data:
        rand_retailer = getMeta(data,'rand_retailer')
      else:
        rand_retailer=random.randint(0,len(config['retailers'])-1)
        json['rand_retailer']=rand_retailer
      data["cat_retailer"]=config['retailers'][rand_retailer]

    #Product Name

    name=createProductName(contents, data["cat_retailer"], gender, color)
    #Product Description
    #Selling Points

    minConfidence = config["global"]["AWSMinConfidence"]
    maxConfidence = config["global"]["AWSMaxConfidence"]
    client=boto3.client('rekognition', 
    aws_access_key_id=settings['awsAccessKeyId'],
    aws_secret_access_key=settings['awsSecretAccessKey'],
    region_name=settings['awsRegionName']
    )
    data = open(photo, "rb").read()

    response = client.detect_labels(Image={'Bytes':data},
        MaxLabels=20)

    #print('Detected labels for ' + photo) 
    labels=[]
    labelsMax=[]
    catMax=[]
    cat=[]
    rec={}
    #print (response)
    rec['AWS']=response['Labels']
    for label in response['Labels']:
        #print ("Label: " + label['Name'])
        #print ("Confidence: " + str(label['Confidence']))
        if int(label['Confidence'])>minConfidence:
            labels.append(label['Name'])
            cat.append(addParents(label['Parents'],label['Name']))
        if int(label['Confidence'])>maxConfidence:
            labelsMax.append(label['Name'])
            catMax.append(addParents(label['Parents'],label['Name']))
              #labels.append('P:'+parent['Name'])
        #    print ("   " + parent['Name'])
        #print ("Instances:")
        for instance in label['Instances']:
          #print (instance)
          cat.append(addParents(label['Parents'],label['Name']))

    labels = list(set(labels))
    cat = list(set(cat))
    labelsMax = list(set(labelsMax))
    catMax = list(set(catMax))
    print (cat)
    #print (labels)
    rec['categories']=cat
    rec['labels']=labels
    rec['max_categories']=catMax
    rec['max_labels']=labelsMax
    return rec

def loadJson(photo):
  newrecord={}
  photo = photo.replace('.jpg','.json')
  with open(photo, "r",encoding='utf-8') as fh:
        text = fh.read()
        newrecord = json.loads(text)
  return newrecord

def updateJson(photo, updaterec,loadonly):
  photo = photo.replace('.jpg','.json')
  newimages=[]
  if ('images' in updaterec):
    newimages=updaterec['images']
  try:
      with open(photo, "r",encoding='utf-8') as fh:
        text = fh.read()
        newrecord = json.loads(text)
        updaterec.update(newrecord)
      #print (updaterecorig)
      for img in newimages:
        if not img in updaterec['images']:
          updaterec['images'].append(img)
        #break
      if not loadonly:
       with open(photo, "w", encoding='utf-8') as handler:
        text = json.dumps(updaterec, ensure_ascii=True)
        handler.write(text)
  except:
    pass
  return updaterec

def process(image, settings, skipAlreadyDone, config):
    rec={}
    data = loadJson(image)
    if 'OpenAI' in data and skipAlreadyDone:
        print("Skipping, already done in openAI")
    else:
      #take the _0 photo
      rec=detect_openai(data, settings, config)
      data= updateJson(image,rec, False)

    return data

def getMeta(data, field):
  if field in data:
    return data[field]
  else:
    return ''


def processImages(filename):
  #Get All files
  settings, config = loadConfiguration(filename)

  allCat=[]
  allColor=[]
  report={}
  total=0
  report['Total'] = 0
  
  catnr = 0
  while catnr<len(config['categories']):
    cat = config['categories'][catnr]['searchFor']
    urls = glob.glob(config['global']['fileLocation']+cat+'\\*.jpg',recursive=True)
    print ("Path: "+config['global']['fileLocation']+cat+'\\*.jpg')
    for image in urls:
      if ('_' not in image):
        print (str(total)+" =>"+image)
        total=total+1
        data = process(image,settings, config['global']['OpenAISkipExisting'], config)

    catnr=catnr+1
  
  print("We are done!\n")
  print("Processed: "+str(total)+" results\n")
  print("NEXT STEP: Push: 5_pushToCatalog")
  
#print(rgbToColor('c6bcb1'))
#print(rgbToColor('c4bdb9'))

#main('../images/dress/blue/1375840.jpg')
#main('../images/dress/blue/767990.jpg')
#main('../images/dress/blue/1569178.jpg')
try:
  fileconfig = sys.argv[1]
  processImages(fileconfig)
except Exception as e: 
  print(e)
  traceback.print_exception(*sys.exc_info())
  print ("Specify configuration json (like config.json) on startup")
