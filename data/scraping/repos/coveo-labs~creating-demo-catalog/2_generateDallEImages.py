from tokenize import String
import sys
import copy
import os
import json
import traceback
import re
import openai
import csv
import random
import pandas as pd
from pathlib import Path
import itertools
import utils
import time
from dalle import Dalle2
import urllib.request
from datetime import datetime
import glob

ALL_PARTS=[]
PARTS=[]
PARTS_POINTER = 0
HTML=''
ADD_DALLE=True

PRODUCTS = utils.json_load('../outputs/dalle.json')
DALLE = Dalle2("sess-OJMr25")

def dir_empty(dir_path):
    my_file = Path(dir_path)
    if my_file.is_dir():
      return not next(os.scandir(dir_path), None)
    else:
      return True

def getImages(query, dir, name):
  global HTML
  global DALLE
  size='300px'
  result = False
  #execute query in dalle
  print("Directory: "+'..\\images\\'+dir)
  if ( dir_empty('..\\images\\'+dir)):
    if ADD_DALLE:
      print("Generating DALLE: "+query)
      generations = DALLE.generate(query)
      result = True
      #wait for images, store them in dir
      for generation in generations:
        image_url = generation["generation"]["image_path"]
        file_path = Path('..\\images\\'+dir, generation['id']).with_suffix('.webp')
        urllib.request.urlretrieve(image_url, file_path)
        image = "<img width="+size+"  src='"+str(file_path)+"'></img>"
        HTML+= "<tr><td>"+name+'</td><td>'+image+'</td></tr>'
    else:
      print("NOT Generating DALLE: "+query)
  else:
    #we already done it, add them to the HTML
    print("DALLE already executed, getting images only: "+query)
    jsons = glob.glob('..\\images\\'+dir+'\\*.*',recursive=True)
    for filename in jsons:
      image = "<img width="+size+" src='"+str(filename)+"'></img>"
      HTML+= "<tr><td>"+name+'</td><td>'+image+'</td></tr>'
  return result

def process():
  # Get All files
  global HTML
  moveOnNextDir=True
  HTML+='<html><head></head><body><table border=1>'
  total = 0
  previousCats={}
  for product in PRODUCTS:
    query = product['dalle']
    dir = product['dir']
    name = product['name']
    currentCat = '\\'.join(dir.split('\\')[:len(dir.split('\\'))-1])
    #print("Current Category: "+currentCat)
    if currentCat in previousCats:
      #do nothing
      if previousCats[currentCat]>=50:
        pass
      else:
        print("Current Category (next): "+currentCat)
        print("Processing (next)...")
        result = getImages(query, dir,name)
        if result and moveOnNextDir:
          previousCats[currentCat]=previousCats[currentCat]+1
        if not result and moveOnNextDir:
          pass
        if result and not moveOnNextDir:
            previousCats[currentCat]=previousCats[currentCat]+1
        if not result and not moveOnNextDir:
            previousCats[currentCat]=previousCats[currentCat]+1
    else:
      print("Current Category: "+currentCat)
      previousCats[currentCat]=1
      print("Processing...")
      result = getImages(query, dir,name)
      #break
    total +=1

  HTML+='</table></body></html>'
  now = str(datetime.now()).replace(':','_').replace('.','_')
  with open(Path('../outputs/dalle_GEN_'+str(now)+'.html'), 'w', encoding='utf-8') as f:
        f.write(HTML)
      
  print("We are done!\n")
  
  print("All products   : " + str(total))
  


try:
  process()
except Exception as e:
  print(e)
  traceback.print_exception(*sys.exc_info())
