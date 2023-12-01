import json
from datasets import load_dataset
import textwrap
import openai
import os
import pinecone
from sentence_transformers import SentenceTransformer
import torch
from tqdm.auto import tqdm
import pandas
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Table, MetaData
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session
import re
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
Base = declarative_base()

class Category(Base):
    __tablename__ = 'categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    en = Column(String)
    topic_id = Column(Integer)
    slug = Column(String)
    shop_id = Column(Integer)
    

class ShopCategory(Base):
    __tablename__ = 'shop_categories'
    
    id = Column(Integer, primary_key=True)
    shop_id = Column(Integer)
    name = Column(String)
    category_id = Column(Integer, ForeignKey('categories.id'))
    parent_id = Column(Integer)
    custom = Column(Boolean)
    category = relationship("Category")

from sqlalchemy.orm import Session
import re

class Shop:
    def __init__(self, session: Session, shop_id: int):
        self.session = session
        self.shop_id = shop_id
        self.custom_counter = 0

   

    def create_shop_categories(self, categories, parent_id=None):
        custom = False
        for raw_name, children in categories.items():
            name = re.sub('^[\d\.]+\s*', '', raw_name)
            name = re.sub('\s*-.*$', '', name)

            existing_category = self.session.query(Category).filter(Category.en == name, Category.topic_id != 11).first()

            if not existing_category:
                try:
                    existing_category = Category(name=name, topic_id=11, slug='custom', shop_id=self.shop_id, en=name)
                    self.session.add(existing_category)
                    self.session.flush()  # Flush the session to get the id of the new Category
                    custom = True
                    self.custom_counter += 1
                except Exception as e:
                    print(f"Error creating Category: {e}")
                    return  # Return from function if the Category can't be created

            try:
                shop_category = ShopCategory(shop_id=self.shop_id, name=existing_category.en, 
                                            category_id=existing_category.id, parent_id=parent_id, custom=custom)
                self.session.add(shop_category)
                self.session.flush()  # Flush the session to get the id of the new ShopCategory
                custom = False
            except Exception as e:
                print(f"Error creating ShopCategory: {e}")
                return  # Return from function if the ShopCategory can't be created

            if isinstance(children, dict) and children:
                self.create_shop_categories(children, shop_category.id)
                
        self.session.commit()




    def parse_categories(self, categories_list):
        if not categories_list:
            return []

        lines = categories_list.split("\n")
        lines = [line.lstrip() for line in lines]
        tree = {}
        path = [tree]

        for line in lines:
            depth = len(re.findall('\d', line))
            name = line.strip()
            name = re.sub('^[\d\.]+\s*', '', name)
            name = re.sub('\s*-.*$', '', name)
            name = re.sub('\sund.*$', '', name)
            name = re.sub('\s&.*$', '', name)

            if not name:
                continue
            if depth == 0:
                path = [tree]
            else:
                path = path[:depth]

            location = path[-1]
            location[name] = {}
            path.append(location[name])

        return tree

topic  = input("\nTopic: ")
k = input("How many proposals do you want from Pinecone (Enter for 100) ? ")
# ideas = input("Shall i stuff the tree with some own ideas as custom categores (may be deleted later)(y/N)?")
shop_id = input("Whats the id of your shop ? ")
# ideas_yes = 0
# if (ideas == "y" or ideas == "yes"):
#     ideas_yes=1

engine = create_engine('mysql+mysqlconnector://forge:559MKSy2FkgOWP280JTZ@localhost/ubernet_affiliateshop')
metadata = MetaData()
Session = sessionmaker(bind=engine)

session = Session()

exists = session.query(ShopCategory.shop_id).filter_by(shop_id=shop_id).first() is not None
if exists:
    print("Shop "+shop_id+" already exists. Remove its categories first")
    delete = input("Or shall i remove it ? (Be 100% sure!!!!!!), type 'delete': ")
    if (delete=="delete"):
        shop_categories = Table('shop_categories', metadata)
        # find shop_categories entries where shop_id=1 and delete them
        session.query(ShopCategory).filter(ShopCategory.shop_id == shop_id).delete()
    else: sys.exit()   

if k=="":
    k = 100
else: k = int(k)


def wrap_text(text):
    wrapper = textwrap.TextWrapper(width=140)
    return wrapper.wrap(text=text)

query = ""

print("Python: 'Hey ChatGPT, i need some ideas about "+topic+"'")

if __name__ == '__main__':
    openai.api_key = os.environ.get("OPENAI_API_KEY") 
    completion = openai.Completion.create(max_tokens=800, engine="text-davinci-003", prompt="Can you give me around 20 ideas for the main categories of my shop about "+topic)
    lines = wrap_text(completion.choices[0].text)
    query = topic + "("+(", ".join(lines))+")"

print("ChatGPT: 'Piny ? There ? I have some cool ideas for you in place!'")
# print(lines)

file = open('categories.json')
dataset = json.load(file)
dataset

cats = []
cats =  dataset['items']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # # if device != 'cuda':
# # #     print(f"You are using {device}. This is much slower than using "
# # #           "a CUDA-enabled GPU. If on Colab you can change this by "
# # #           "clicking Runtime > Change runtime type > GPU.")

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
model

xq = model.encode(query)
xq.shape

# get api key from app.pinecone.io
PINECONE_API_KEY = "e7980b1a-dadb-4ae4-a97a-a7a73a1af9ff"
# find your environment next to the api key in pinecone console
PINECONE_ENV = "asia-southeast1-gcp-free"

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = 'categories'

# # only create index if it doesn't exist
#if index_name not in pinecone.list_indexes():
#    pinecone.create_index(
#        name=index_name,
#         dimension=model.get_sentence_embedding_dimension(),
#         metric='cosine'
#     )

# # now connect to the index
index = pinecone.GRPCIndex(index_name)



# batch_size = 128
# bla = tqdm(range(0, len(cats), batch_size))

# for i in bla:
#     # find end of batch
#     i_end = min(i+batch_size, len(cats))
#     # create IDs batch
#     ids = [str(x) for x in range(i, i_end)]
#     # create metadata batch
#     metadatas = [{'text': text} for text in cats[i:i_end]]
#     # create embeddings
#     xc = model.encode(cats[i:i_end])
#     # create records list for upsert
#     records = zip(ids, xc, metadatas)
#     #upsert to Pinecone
#     index.upsert(vectors=records)
#      # # check number of records in the index
#     index.describe_index_stats()



# create the query vector
xq = model.encode(query).tolist()

# now query
xc = index.query(xq, top_k=k, include_metadata=True)
xc

outputCategories = []

for result in xc['matches']:
    outputCategories.append(result['metadata']['text'])

print("Pinecone: 'Thanks Chatty ! Here are your "+str(len(outputCategories))+" ideas'")
# print(outputCategories)


prompt = "Create me a detailed category structure with multiple root categories. use max 8 root categories and max depth 3.use clearly findable category names. rather longer than shorter. Give me the list as a reversly numbered tree (e.g. 1, 1.1, 1.2, 2, 2.1, 2.2, 2.2.1 ..).ident the categories properly ! All Categories must have a number. No descriptions or subordinate clauses."
# if (ideas_yes == 1):
#     prompt += ".add maximum 10 category-ideas to the whole tree. add a prefix C- only to these categories. not the other ones."
prompt += "Do not use category names that consist of multiple words: "+", ".join(outputCategories)

print("ChatGPT: 'I create the shop structure now from piny's cool ideas.thanks for this!'")

message = [
    {"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": prompt}, 
]

openai.api_key = os.environ.get("OPENAI_API_KEY")
completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages = message,
    temperature=0.2,
    max_tokens=4000,
    frequency_penalty=0.0
)


text = completion.choices[0].message.content;


#print("The Result:")/
#print(text)

print("Python: Now i create the shop for you"); 
shop = Shop(session, shop_id=shop_id)
tree = shop.parse_categories(text)
shop.create_shop_categories(tree)

print("Python: 'Your Tree has been created !'")

print("\n\nPimnecone: 'Now i check which advertisers on AWIN match your shop..be patient'\n\n")

pinecone.init(
    api_key="fd769e8c-27ca-42ee-8ec1-55c2accdcead",
    environment="us-west1-gcp-free"
)

index_name = 'awin'
# # ngw connect to the index
index = pinecone.GRPCIndex(index_name)
file = open('awin_14_12_2022.json')
dataset = json.load(file)
dataset

advertisers = []
advertisers =  dataset['items']

# print("First we insert all advertisers from awin to pinecone")
# advertisers = pandas.Series(advertisers)
# batch_size = 128
# bla = tqdm(range(0, len(advertisers), batch_size))

# for i in bla:
#     # find end of batch
#     i_end = min(i+batch_size, len(advertisers))
#     # create IDs batch
#     ids = [str(x) for x in range(i, i_end)]
#     # create metadata batch
#     metadatas = [{'text': json.dumps(advertiser)} for advertiser in advertisers[i:i_end]]
#     # create embeddings
#     xc = [model.encode(json.dumps(advertiser)).tolist() for advertiser in advertisers[i:i_end]]
#     # create records list for upsert
#     records = list(zip(ids, xc, metadatas))
#     # upsert to Pinecone
#     index.upsert(vectors=records)

query = topic

# create the query vector
xq = model.encode(query).tolist()

# now query
xc = index.query(xq, top_k=k, include_metadata=True)
xc

outputAdvertisers = []



for result in xc['matches']:
    # Load the JSON string into a Python dictionary
    advertiser = json.loads(result['metadata']['text'])
    # Print the programmeName and the score
    if float(result['score']) > 0.20:
        print(f"Programme Name: {advertiser['programmeName']}, Score: {round(result['score'], 2)}")
