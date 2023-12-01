import os
import json
import re
import sys
import base64
import operator
import traceback
from webcolors import (
    CSS3_NAMES_TO_HEX,
    hex_to_rgb,
)
import glob
import random
import openai

currentJson=[]

P_ENGINE = 'curie-instruct-beta-v2'



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


def createProductName(keywords, brand, gender, color, temp=0.9, length=100):
  print("Starting Product Name for:")
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
  print("Starting Product Description for:")
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
  print("Starting Product Selling for:")
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
  print("Starting Product Features for:")
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
  print("Starting Product Article for:")
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
  openai.api_key = settings['openaiApiKey']

  return settings, config

def basicColors():
  mainColors = ["black","silver","gray","white","maroon","red","purple","fuchsia","green","lime","olive","yellow","navy","blue","teal","aqua"]
  return mainColors

def normalizeColors(color):
   mainColors = basicColors() 
   for col in mainColors:
     if col in color:
       color += ' '+col
   return color.title()


def removeNumbers(input):
  r=re.compile(r'\d')
  output = r.sub('', input)
  return output

def getAllColors():
    css3_db = CSS3_NAMES_TO_HEX
    names = []
    rgb_values = []
    for color_name, color_hex in css3_db.items():
        names.append(color_name)
    for color_name in basicColors():
      names.append(color_name)
    return names

def checkCat(data,config,allcolors):
  #Is Person|Man in there
  #Is Person|Female in there
  gender='Men;Women'
  if 'categories' in data:
    if 'Person|Man' in data['categories']:
      gender='Men'
    if 'Person|Female' in data['categories']:
      gender='Women'
  title=getMeta(data,'category')
  author = getMeta(data,'photographer')
  if author:
    #only first name
    author = author.split(' ')[0]
    title = title +' by '+author
    title = title.title()
  url = getMeta(data,'url')
  #color in url is leading
  colorurl = ''
  for col in allcolors:
    if col in url:
      colorurl = col
  if colorurl:
    data['bestcolor']=colorurl.title()

  #title is also based upon url
  titleurl = url
  titleurl = titleurl.replace('https://www.pexels.com/photo/','')
  #print (titleurl)
  #check for gender in title
  if 'woman' in titleurl or 'women' in titleurl:
    gender='Women'
  else:
    if 'men-' in titleurl or 'man-' in titleurl:
      gender='Men'
  if config['useGender']:
    if not config['defaultGender']=='':
      gender = config['defaultGender']
  else:
    gender=''
  #print (gender+' = '+colorurl)
  titleurl = removeNumbers(titleurl.replace('-',' ').replace('/',' ')).title()
  #/women-s-yellow-floral-v-neck-top
  data['titledescr']=titleurl
  data['gender']=gender
  return data



def updateJson(photo, data):
  photo = photo.replace('.jpg','.json')
  #newimages=[]
  #if ('images' in updaterec):
  #  newimages=updaterec['images']
  try:
      # with open(photo, "r",encoding='utf-8') as fh:
      #   text = fh.read()
      #   newrecord = json.loads(text)
      #   updaterec.update(newrecord)
      # #print (updaterecorig)
      # for img in newimages:
      #   if not img in updaterec['images']:
      #     updaterec['images'].append(img)
      #   #break
      # if not loadonly:
       with open(photo, "w", encoding='utf-8') as handler:
        text = json.dumps(data, ensure_ascii=True)
        handler.write(text)
  except:
    pass
  return 



def loadJson(photo):
  newrecord={}
  photo = photo.replace('.jpg','.json')
  try:
      with open(photo, "r",encoding='utf-8') as fh:
        text = fh.read()
        newrecord = json.loads(text)
  except:
    pass
  return newrecord

def cleanLabels(data):
  toremove=["Person","Human"]
  newdata=[]
  for item in data:
    if item not in toremove:
      newdata.append(item)
  return newdata

def fixImages(images, config, baseurl, fileloc):
  newimages=[]
  for image in images:
    #..\\images\\hat\\601168.jpg
    newfile = image.replace(fileloc,baseurl)
    newfile = newfile.replace('..\\images\\',baseurl)
    newfile = newfile.replace('../images/',baseurl)
    newfile = newfile.replace('\\','/')
    newimages.append(newfile)
  return newimages

def getMeta(data, field):
  if field in data:
    return data[field]
  else:
    return ''


def createCategories(json, config):
  categories=[]
  if config['useGender']:
    genders=' and '.join(getMeta(json,'gender').split(';'))
    categories.append(genders)
    for cat in getMeta(config,'categoryPath').split('|'):
      categories.append(cat)
  else:
    for cat in getMeta(config,'categoryPath').split('|'):
      categories.append(cat)

  #categories = list(set(categories))

  return categories


def createCategoriesPaths(categories):
  catpath=''
  catpaths=[]
  for cat in categories:
     if catpath=='':
       catpath=cat #man
       catpaths.append(catpath)
     else:
       catpath=catpath+'|'+cat
       catpaths.append(catpath)
  #catpaths = list(set(catpaths))
  return catpaths

def createCategoriesSlug(categories):
  slug=[]
  catpath=''
  for cat in categories:
    cat=cat.lower().replace(' ','-')
    if catpath=='':
      catpath=cat #man
      slug.append(catpath)
    else:
      catpath=catpath+'/'+cat
      slug.append(catpath)
  
  #slug = list(set(catpaths))

  return slug

def createFacets(rec, facets, variant):
  facetData = ''
  createVariant=False
  for facet in facets:
    if facet['variant']==variant:
      facetname='cat_'+facet['name'].replace(' ','_').lower()
      if not facet['useWhenPropertyValue']=="":
        #then do not use random values, but the values set
        if 'default' in facet and facetname not in rec:
          facetvalue = facet['default']
        #check if useWhenPropertyValue is in cat_properties
        if 'cat_properties' in rec:
          #print (rec)
          if facet['useWhenPropertyValue'] in rec['cat_properties']:
            facetvalue=facet['values']
      else:
        values = facet['values'].split(';')
        facetvalue=values[random.randint(0,len(values)-1)]

      rec[facetname]=facetvalue
      facetData = facetData+' '+facetvalue
      createVariant=True

  return rec, facetData, createVariant


def createVariants(json, productid, facets):
  rec={}
  facetData = ''
  rec['cat_properties'] = json['cat_properties']
  rec, facetData, createVariant = createFacets(rec, facets, True)

  sku = productid+'_'+facetData.replace(' ','_').replace('/','_')
  rec["DocumentId"]= json['DocumentId']+'?sku='+sku
  rec["DocumentType"]="Variant"
  rec["FileExtension"]=".txt"
  rec["ObjectType"]="Variant"
  rec['data']=json['data']+' '+facetData
  rec['ec_name']=json['title']#getMeta(json,'titledescr')+facetData
  rec["title"]=json['title']#rec['ec_name']
  rec['ec_title']=rec['ec_name']
  rec["ec_product_id"]=productid
  rec["ec_variant_sku"]=sku
  rec["permanentid"]=sku
  saveJson(rec)

def process(image,json, allcolors, config, fileloc, baseurl, baseurlimages,child, UseMaxLabels,childnr):
  #create final json
  print("Processing: "+str(json['id']))
  json = checkCat(json,config,allcolors)
  rec={}
  color_hex = getMeta(json, 'man_hexcolor')
  color = getMeta(json, 'man_color')
  productid= json['category']+str(json['id'])+'_'+str(color_hex)
  images = getMeta(json,'images')
  if child:
    print("Processing CHILD: "+str(json['id']))
    color_hex = getMeta(child, 'colorhex')
    color = getMeta(child, 'color')
    productid= json['category']+str(json['id'])+'_'+str(color_hex)
    images = getMeta(child,'images')


  images=fixImages(images, config, baseurlimages, fileloc)

  rec['DocumentId']=baseurl+json['category']+'/'+productid
  json['DocumentId']=rec['DocumentId']
  rec['ec_item_group_id']=json['category']+str(json['id'])
  rec["DocumentType"]="Product"
  rec["FileExtension"]=".html"
  rec["ObjectType"]="Product"
  if UseMaxLabels:
    rec["cat_attributes"]=cleanLabels(getMeta(json,'max_labels'))
  else:
    rec["cat_attributes"]=cleanLabels(getMeta(json,'labels'))
  # categories=[]
  # for cat in getMeta(json,'categories'):
  #   if 'Clothing|' in cat:
  #     cleancat = cat.replace('Clothing|','')
  #     if cleancat not in categories:
  #       categories.append(cleancat)
  # categories.sort()
  # categoriesclean=[]
  # for category in categories:
  #   for cat in category.split('|'):
  #     if cat not in categoriesclean:
  #       categoriesclean.append(cat)

  rec["cat_categories"]=createCategories(json,config)
  rec["cat_slug"]=createCategoriesSlug(rec["cat_categories"])
  rec["ec_category"]=createCategoriesPaths(rec["cat_categories"])
  rec["cat_color"]=color
  rec["cat_colorhex"]=str(color_hex)
  rec["cat_gender"]=getMeta(json,'gender')
  rec["cat_mrp"]=config['avgPrice']+random.randint(0,int(config['avgPrice']/10))+0.75
  if ('cat_retailer' not in json):
    if 'rand_retailer' in json:
      rand_retailer = getMeta(json,'rand_retailer')
    else:
      rand_retailer=random.randint(0,len(config['retailers'])-1)
      json['rand_retailer']=rand_retailer
    rec["cat_retailer"]=config['retailers'][rand_retailer]
  colors = normalizeColors(color)
  labels = ""
  if UseMaxLabels:
    rec["data"]=" ".join(getMeta(json,'max_labels'))+' '+getMeta(config,'categoryPath').replace('|',' ')+' '+getMeta(json,'titledescr')+' '+colors
    labels = getMeta(json,'category')+", "+", ".join(getMeta(json,'max_labels'))
  else:
    rec["data"]=" ".join(getMeta(json,'labels'))+' '+getMeta(config,'categoryPath').replace('|',' ')+' '+getMeta(json,'titledescr')+' '+colors
    labels = getMeta(json,'category')+", "+", ".join(getMeta(json,'labels'))

  ################ from OPENAI
  #Product Name
  aititle=getMeta(json,'aititle')
  sell=''
  #aititle=''
  description=''
  if not aititle:
    title=createProductName(labels, rec["cat_retailer"], rec["cat_gender"], colors)
    if not title:
      title = color+' '+json["category"]+' by '+getMeta(json,'photographer')
      title = title.title()
    #Product Description
    if not child:
      description = createProductDescription(title)
      if not description:
        description = rec["data"]
      #Selling Points
      #sell=createProductSelling(description)
    if child:
      json['childs'][childnr]['aititle'] = title
    else:
      json['aititle'] = title
      json['description'] = description
      json['sell']= sell
    updateJson(image, json)
  else:
    if child:
      title = getMeta(child, 'aititle')
      #title = ''
      if not title:
        title=createProductName(labels, rec["cat_retailer"], rec["cat_gender"], colors)
        if not title:
          title = color+' '+json["category"]+' by '+getMeta(json,'photographer')
          title = title.title()
        json['childs'][childnr]['aititle'] = title
        updateJson(image, json)


      description = ''
      sell = ''
    else:
      title = getMeta(json, 'aititle')
      description = getMeta(json, 'description')
      sell = getMeta(json, 'sell')

  rec['data']=rec['data']+' '+sell
  json['data']=rec['data']
  rec["ec_brand"]=rec["cat_retailer"]
  rec["ec_brand_name"]=rec["cat_retailer"]
  rec["ec_description"]=description
  rec["ec_images"]=images
  rec["ec_name"]=title#getMeta(json,'titledescr')
  rec["title"]=title#rec['ec_name']
  json['title']=title
  rec["ec_price"]=rec["cat_mrp"]
  rec["ec_product_id"]=productid
  rec["permanentid"]=rec['ec_product_id']
  rec["pexel_url"]=json['url']
  json['categories'].sort()
  json['max_categories'].sort()
  if UseMaxLabels:
    rec["cat_properties"]=json['max_categories']
  else:
    rec["cat_properties"]=json['categories']

  json['cat_properties']=rec['cat_properties']

  #add facets
  createVariant = False
  rec, facetData, createVariant = createFacets(rec, config['facets'], False)
  rec['data']+= ' '+facetData

  saveJson(rec)
  if createVariant:
    createVariants(json, productid, config['facets'])

  return json


def storeJson(jsond,config):
  try:
       filename = config['global']['catalogJsonFile']
       directory = os.path.dirname(os.path.abspath(filename))
       if (not os.path.isdir(directory)):
        os.makedirs(directory) 
       with open(filename, "w", encoding='utf-8') as handler:
         text = json.dumps(jsond, ensure_ascii=True)
         handler.write(text)
  except:
    print("Error storing json")
    pass
  return 

def saveJson(json):
  global currentJson
  currentJson.append(json)
  return

def processImages(filename):
  global currentJson
  
  allcolors=getAllColors()

  total=0
  totalf=0
  totalchild=0
  uniqueproducts=0
  #per category
  k=0
  settings, config = loadConfiguration(filename)
  
  catnr = 0
  while catnr<len(config['categories']):
    totalnochild=0
    totalwithchild=0
    cat = config['categories'][catnr]['searchFor']
    urls = glob.glob(config['global']['fileLocation']+cat+'\\*.jpg',recursive=True)
    print ("Path: "+config['global']['fileLocation']+cat+'\\*.jpg')
    imgnr=0
    prevnr=-1
    while imgnr<len(urls):
      image = urls[imgnr]
      #print (image)
      if ('_' not in image):
        currentFile = image
        #print (str(total)+" =>"+image)
        json=loadJson(image)
        #print (json)
        #if already done
        #if total>20:
        #  break
        if 'colorxy' in json:
          total=total+1
          kids=False
          childnr=0
          json=process(image,json,allcolors,config['categories'][catnr], config['global']['fileLocation'],config["global"]["baseUrl"],config["global"]["baseUrlImages"],None,config["global"]["UseMaxLabels"],0)
          if 'childs' in json:
            for child in json['childs']:
                json=process(image,json, allcolors, config['categories'][catnr],config['global']['fileLocation'],config["global"]["baseUrl"],config["global"]["baseUrlImages"],child,config["global"]["UseMaxLabels"],childnr)
                childnr=childnr+1
                totalchild=totalchild+1
                kids=True
          if kids:
            totalwithchild += 1
          else:
            totalnochild +=1
          imgnr = imgnr+1
        else:
          imgnr = imgnr+1
          totalf=totalf+1
      else:
        imgnr = imgnr+1
    #catnr = catnr+1

    print("CATEGORY: "+cat)
    print("Processed with Childs   : "+str(totalwithchild)+" results\n")
    print("Processed without Childs: "+str(totalnochild)+" results\n")
    print("==========================================\n")
    catnr=catnr+1

  #print (currentJson)
  storeJson(currentJson,config)  
  print("We are done!\n")
  print("Processed       : "+str(total)+" results\n")
  print("Processed BAD      : "+str(totalf)+" results\n")
  print("Processed Childs: "+str(totalchild)+" results\n")
    

print("NEXT STEP: 5. Push Catalog.")


try:
  fileconfig = sys.argv[1]
  processImages(fileconfig)
except Exception as e: 
  print(e)
  traceback.print_exception(*sys.exc_info())
  print ("Specify configuration json (like config.json) on startup")
