from openai import OpenAI
from dotenv import load_dotenv # pip install python-dotenv
from pyzotero import zotero
load_dotenv()
import os

zotero_api_key = os.environ.get("ZOTERO_API_KEY")
zotero_user_id = os.environ.get("ZOTERO_USER_ID")
zot = zotero.Zotero(zotero_user_id, 'user', zotero_api_key)

col = zot.collection('IRF6FT7U')
items = zot.collection_items('IRF6FT7U')

pdf_dir = './pdfs'
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)

for item in items:
    if 'contentType' in item['data'] and item['data']['contentType'] == 'application/pdf':
        filepath = pdf_dir + '/' + item['data']['filename']
        if not os.path.exists(filepath):
            zot.dump(item['data']['key'],item['data']['filename'],pdf_dir)
        print(item['data'])

#openai_api_key=os.environ.get("OPENAI_API_KEY")
#print("api_key",openai_api_key)

client = OpenAI()

model_list = client.models.list()
for model in model_list:
    #print("id:", model.id)
    pass
