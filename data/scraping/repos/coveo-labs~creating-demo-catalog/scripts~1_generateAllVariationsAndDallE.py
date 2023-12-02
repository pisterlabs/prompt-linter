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

ALL_PARTS=[]
PARTS=[]
PARTS_POINTER = 0
BASE_DALLE= 'a photo with a [TITLE] in the middle, surrounded by a white background, vray, full shot'
#a photo with a red life jacket in the middle surrounded by white background, vray, full shot
df = pd.read_excel(Path('../Barca - Brand Detail.xlsx'), sheet_name='Barca Sports Taxonomy', na_filter=False)

def readCSV():
  global ALL_PARTS
  ALL_PARTS = df.to_dict(orient='records')
  level1 = ''
  level2 = ''
  level3 = ''
  dalle = ''
  id=''
  # Fix the parts, because level1, level2, level3 might not be there
  for part in ALL_PARTS:
    #print (part)
    #part['DallE name'] = part['Category Level 1']
    if part['DallE name'] == '':
      part['DallE name'] = dalle
    else:
      dalle = part['DallE name'].strip()
    if part['ID'] == '':
      part['ID'] = id
    else:
      id = part['ID']
    part['Category Level 1'] = part['Category Level 1']
    if part['Category Level 1'] == '':
      part['Category Level 1'] = level1
    else:
      level1 = part['Category Level 1'].strip()
    if part['Category Level 2'] == '':
      part['Category Level 2'] = level2
    else:
      if level2 != part['Category Level 2'].strip():
        level3 = ''
      level2 = part['Category Level 2'].strip()
    if part['Category Level 3'] == '':
      part['Category Level 3'] = level3
    else:
      level3 = part['Category Level 3'].strip()
    field = part['Fields'].strip()
    part['Fields'] = field#.lower()


def createBigPartsList():
  global PARTS
  PARTS = []
  currentPart = {}
  for part in ALL_PARTS:
    #print (part)
    if (part['Fields'] == 'ec_Price'):
      # this is the last record
      currentPart[part['Fields']] = part['Values']
      currentPart['DallE name'] = part['DallE name'].strip()
      currentPart['ID'] = part['ID']
      currentPart['Category Level 1'] = part['Category Level 1'].strip()
      currentPart['Category Level 2'] = part['Category Level 2'].strip()
      currentPart['Category Level 3'] = part['Category Level 3'].strip()
      PARTS.append(json.dumps(currentPart))
      currentPart = {}
    else:
      currentPart[part['Fields']] = part['Values'];#.strip()


def getAllValues(record, variant_fields):
  new_record = {}
  #record = addVariants(record, variant_fields)
  keys = record.keys()

  for key in keys:
    if (isinstance(record[key], str) and ';' in record[key]):
        # multiple values
      values = []
      curvalues = record[key].split(';')
      for val in curvalues:
        if (val.strip() != ''):
          values.append((val,))
      new_record[key] = values
      print(new_record[key])
    else:
      if isinstance(record[key], str):
        new_record[key] = (record[key],)
      else:
        new_record[key] = (str(record[key]),)
      print(new_record[key])
  keys = new_record.keys()
  values = (new_record[key] for key in keys)
  #print ("GetAllValues")
  #print (new_record)
  combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
  # clean up the combinations
  records = []
  for combi in combinations:
    new_record = {}
    for key in combi.keys():
      if isinstance(combi[key], tuple):
        new_record[key] = ''.join(combi[key])
      else:
        new_record[key] = combi[key]
      if (isinstance(new_record[key], str)):
        if (new_record[key].upper() == 'YES'):
          new_record[key] = True
        else:
          if (new_record[key].upper() == 'NO'):
            new_record[key] = False

    records.append(new_record)
  return records


def removeSpecialFieldsFromKey(metadata):
  meta = copy.deepcopy(metadata)
  return meta


def createVariantKey(metadata, variant_fields):
  variant_key = ''
  meta = removeSpecialFieldsFromKey(metadata)
  for key in meta.keys():
    if key not in variant_fields:
      variant_key += str(meta[key])
    else:
      # 7variant_key+="1"
      pass

  return variant_key


def doWeHaveVariants(metadata, variant_fields):
  variants = False
  meta = removeSpecialFieldsFromKey(metadata)
  for key in meta.keys():
    if key in variant_fields:
      if (meta[key] != ''):
        variants = True

  return variants


def getNextPartRecord():
  global PARTS_POINTER
  global PARTS
  record = json.loads(PARTS[PARTS_POINTER])
  PARTS_POINTER += 1
  if (PARTS_POINTER >= len(PARTS)):
    PARTS_POINTER = 0

  return record


def createCategoryFolderForImages(parts):
  cur_path = Path('../images')
  for p in parts:
    if p:
      cur_path = cur_path / p.replace('/', '_')
      if not cur_path.exists():
        print(cur_path)
        cur_path.mkdir()

def createRecord(main_record, parent_name, brands, therecord, recid, boats, versions, variant_fields, groupid, we_have_variants, all_names,all_fields):
  # record=[]
  global BASE_DALLE
  record = copy.deepcopy(therecord)
  mainRecord = True
  if groupid == 0:
    # this is a main product
    productid = 'sp_'+f'{recid:07}'
    mainRecord = True
    groupid = productid
    #record['Variation Parent (StockKeepingUnit)'] = ''
    if we_have_variants:
      # pass
      record['Variation AttributeSet'] = 'WithVariants'
  else:
    # This means a variant
    mainRecord = False
    productid = str(groupid)+'_'+f'{recid:05}'
    record['Variation AttributeSet'] = ''
    # variantKeys = getVariantKeys(record, variant_fields)
    # record['Variation Parent (StockKeepingUnit)'] = groupid
    # for index, elem in enumerate(variantKeys):
    #   record['Variation Attribute Name '+str(index+1)] = elem['field']
    #   record['Variation Attribute Value '+str(index+1)] = elem['value']

  record['permanentid'] = productid
  thename = ''
   #print(record)
  id = str(int(float(record['ID'])))
  #print (record)
  thename = record['DallE name']
  #fix the [COLOR] to [COLORS]
  thename = thename.replace('[COLOR]','[COLORS]')
  #replace possible occurences inside thename [COLOR] [TYPE]
  for field in record:
    #fieldname = '['+field.upper()+']'
    if field.lower() not in all_fields:
      all_fields.append(field.lower())
    fieldname = '['+field+']'
    #print (fieldname)
    if fieldname in thename:
      thename = thename.replace(fieldname,record[field])
      id += '_'+record[field]
      #print(thename)

  record['ID']=id.replace(' ','_')
  record['dalle']=BASE_DALLE.replace('[TITLE]',thename)
  cat3 = record.get('Category Level 3', '').strip()
  if cat3:
    record['ec_name'] = thename
    #record['ec_name'] = (record['Category Level 2'] + ' ' + record['Category Level 1'] + ' with ' + cat3).strip()# + ' from ' + record['ec_brand']).strip()
    record['dir']=record['Category Level 1'].strip()+'\\'+record['Category Level 2'].strip()+'\\'+record['Category Level 3'].strip()
  else:
    record['ec_name'] = thename
    #record['ec_name'] = (record['Category Level 2'] + ' ' + record['Category Level 1']).strip() #+ ' from ' + record['ec_brand']).strip()
    record['dir']=record['Category Level 1'].strip()+'\\'+record['Category Level 2'].strip()

  dir = record['dir']+'\\'+record['ID']
  record['dir']=dir
  createCategoryFolderForImages(dir.split('\\'))
 
  record['ec_category'] = utils.build_hierarchy([record['Category Level 1'], record['Category Level 2'], cat3])
  record['Category'] = record['ec_category'][-1].replace('|', '/')
  record['ec_partnumber'] = str(groupid).replace('sp_', 'B')+' '+str(groupid)[-3:]
  record['ec_partnumber_oem'] = str(groupid)[-3:]+str(groupid).replace('sp_', '-')

  record['ec_productid'] = productid
  #print (record)
  record['ec_item_group_id'] = groupid

  # Fields which needs to be at Variant and Product level
  record['ec_sku'] = productid
  record['ec_in_stock'] = 'TRUE'

  return record, all_names, thename, all_fields

def process(filename):
  # Get All files
  random.seed(10)
  #settings, config = loadConfiguration(filename)
  readCSV()
  # readCSVSKU()
  createBigPartsList()
  total = 0
  total_variants = 0
  current_total = 0
  versions = ['1', '1 beta', '2', '2 beta', '3 beta', '3', '1', '1 beta', '2', '2 beta', '3 beta', '3', '1', '1 beta', '2', '2 beta', '3 beta', '3']
  boats = ['Mercury', 'Yamaha', 'Honda', 'Evinrude', 'Suzuki', 'Johnson', 'Tohatsu', 'OMC', 'Chrysler', 'Force', 'Mariner', 'Mercruiser', 'Mercury', 'Nissan', 'Sears']
  variant_fields = ['cat_Diameter', 'cat_Length', 'cat_Thickness', 'ec_Sizes']
  all_parts = []
  all_queries=[]
  all_fields=[]
  all_parts_and_variants = {}
  all_names = []
  file_counter_parts = 1
  main_record = 1
  print("No of ALL parts:")
  print(len(ALL_PARTS))
  print("No of parts:")
  print(len(PARTS))
  # First process all the parts, for each part get all the possible values (all_records)
  # Then create Records for each brand
  for part in PARTS:
    # for every part for every key create records
    record = getNextPartRecord()
    all_records = []
    all_records = getAllValues(record, variant_fields)

    first = True
    this_is_main = True
    groupid = 0
    variant_keys = []
    boat = ''
    version = ''
    # special case for brands, if we use them they will screw up the variants
    brand_counter = 0
    no_of_brands = 0
    brands = []
    current_brand = ''
    #saveLibs()
    for rec in all_records:
      first = True
      groupid = 0
      parent_name = ''
      total = total+1
      current_total += 1
      recid = total+1000
      print(total)

      variant_key = createVariantKey(rec, variant_fields)
      we_have_variants = doWeHaveVariants(rec, variant_fields)
      # if the variant_key is already in the variant_keys, then we need to set the groupid
      if variant_key == '':
        # no variant info so normal product
        groupid = 0
      if we_have_variants and variant_key in variant_keys and variant_key != '':  # and groupid!=0:
        # already got
        this_is_main = False
        print(variant_key+" ==> Already got, this is a variant")
        total_variants += 1
        pass
      else:
        # we do not have it
        this_is_main = True
        print(variant_key+" ==> Main product")
        groupid = 0
        main_record += 1
        if (variant_key != '') and we_have_variants:
          variant_keys.append(variant_key)

      print("Variants: "+str(we_have_variants))
      # this is the main record

      rec_added, all_names, parent_name, all_fields = createRecord(main_record, parent_name, brands, rec, recid, boats, versions, variant_fields, groupid, we_have_variants, all_names, all_fields)
      groupid = rec_added['ec_item_group_id']
      if this_is_main:
        all_queries.append({'name':parent_name,'dalle':rec_added['dalle'],'id':rec_added['ID'],'dir':rec_added['dir']})

      if groupid in all_parts_and_variants:
        # This is a variant record
        all_parts_and_variants[groupid].append(rec_added)
      else:
        # this is the main record
        #thekey = random.randint(0, len(brands)-1)
        #rec_added['Product EcBrand__c'] = brands[thekey]
        all_parts_and_variants[groupid] = [rec_added]
      all_parts.append(rec_added)
      first = False
      # break

    # break
  utils.json_dump(all_parts, Path('../outputs/products.json'), sort_keys=False)
  utils.json_dump(all_queries, Path('../outputs/dalle.json'), sort_keys=False)
  utils.json_dump(all_fields, Path('../outputs/fields.json'), sort_keys=False)


  print("We are done!\n")
  print("Copy all CSV to C:/Test and run Upload.cmd")
  #print("Processed: "+str(total+total_variants+total_skipped)+" records")
  print("All products   : " + str(total))
  print("Main products   : " + str(main_record))
  print("Variants        :  "+str(total_variants))
  print("Dalle           :  "+str(len(all_queries)))


try:
  process('')
except Exception as e:
  print(e)
  traceback.print_exception(*sys.exc_info())
  #print ("Specify configuration json (like config.json) on startup")
