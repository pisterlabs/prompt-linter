from  langchain.graphs.graph_document import GraphDocument, Node, Relationship
import pandas as pd
from creating_graph import *
from langchain.schema import Document
import json
# Creating Nodes --->

path = 'backendPython/neo4j/Placement data.xlsx'
df = pd.read_excel(path, sheet_name='2020-21')
values = {'Selected': df.iloc[:,2].mode() , 'CTC':df.iloc[:, 4].median() , 'CGPA':0}
df.fillna(value=values, inplace=True)

'''
#   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   Id             260 non-null    int64
 1   Company        260 non-null    object
 2   Selected       260 non-null    int64
 3   About company  259 non-null    object
 4   CTC            226 non-null    object
 5   CGPA           226 non-null    float64
 6   JobProfile     260 non-null    object
 7   Venue          260 non-null    object
'''

nodes_list, relation_list = [], []
cols = list(df.columns)
about = {
  'Id': 'id',
  'Company': 'name of the company',
  'Selected': 'number of students selected by the company',
  'CTC': 'the salary offered by the company',
  'CGPA': 'the minimum cgpa required by the company',
  'JobProfile': 'job profile offered by the company',
  'Venue': 'work loction set by the company', 
}
sets = {
    'CGPA' : {},
    'Selected' : {},
    'CTC' :{},
    'JobProfile' :{},
    'Venue' : {},
    # 'Company' :{}
}
edge_description = json.load(open('backendPython/neo4j/edges.json', 'r'))

for index, row in df.iterrows():
  # creating nodes
  list_ = []
  for i, val in enumerate(row):
    if i==0 or i==3:
      continue
    if i==1:
      # every comapny is unique, in terms of (name, profile)
      id = cols[i]+'_'+str(index)
      type = cols[i]
      properties = {
        'name' : val ,
        'profile' : row.iloc[6] ,
        'description' : about[cols[i]] , 
      }
      node = Node(id=id, type=type, properties=properties)
      list_.append(node)
      nodes_list.append(node)
    elif i==7:
      # venue
      venues = val.split('/')
      venues = [venue.strip() for venue in venues]
      for venue in venues:
        if venue in sets[cols[i]]:
          list_.append(sets[cols[i]][venue])
          continue
        id = cols[i]+'_'+str(index)
        type = cols[i]
        properties = {
          'name' : venue ,
          'description' : about[cols[i]] , 
        }
        node = Node(id=id, type=type, properties=properties)
        list_.append(node)
        nodes_list.append(node)
        sets[cols[i]][venue] = node
      
    else:
      if val in sets[cols[i]]:
        list_.append(sets[cols[i]][val])
        continue
      id = cols[i]+'_'+str(index)
      type = cols[i]
      properties = {
        'name' : val ,
        'description' : about[cols[i]] , 
      }
      node = Node(id=id, type=type, properties=properties)
      list_.append(node)
      nodes_list.append(node)
      sets[cols[i]][val] = node


  # defining relations
  for i in range(6):
    for j in range(6):
      if list_[i].type == list_[j].type: 
        continue
      source , target= list_[i] , list_[j]
      type = '{}_to_{}'.format(source.type, target.type)
      properties = {
        'description' : edge_description[type] , 
      }
      relation = Relationship(source=source, target=target, type=type, properties=properties)
      relation_list.append(relation)

print('created nodes and relations')

# Creating GraphDocument --->
source_url = 'https://onedrive.live.com/edit?id=DC99902FB0C3CAF0!1334&resid=DC99902FB0C3CAF0!1334&ithint=file%2cxlsx&authkey=!AG-WcseUis3mRpY&wdo=2'

graph_document = [GraphDocument(nodes=nodes_list, relationships=relation_list, source=Document(page_content=source_url))]

print('created graph document')   
graph.add_graph_documents(graph_documents=graph_document, include_source=True)

print('all completed successfully!! ')