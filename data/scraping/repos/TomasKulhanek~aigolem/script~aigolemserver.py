from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI,File
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

client = OpenAI()
#of = client.files.create(
#  file=Path("medidetect-test-utf8.csv"),
#  purpose="assistants"
#)


#print(of)
#print("fileid:"+of.id);

#AIGOLEMBASE = 'RA: bezv. DM v rodině - babička ve staří  PA: ochutnávač/ka  OA: DM 1.typu od 22.2.2018, počátek s typic. příznaky, DKA, léčba intenzifikovaným inzulínovým režimem  komp. - HbA1c 88mmol/mol. Zhubnul 6kg.  Účastnil se studie Diagnode, kde úplná remise cca 1,5 roku po podávání protilátky. Poté včas nenasadil dekkvátní dávky inzulínu a hospitalizován s poč. ketoacidosou.  st.p. hernioplastice 2001  AT: konzumace alkoholu:  přílež.  kouření: 20/d    EA: 2 dávky očkování proti COVID-19, poslední 6/2021, COVID+ 1/2022';
AIGOLEMBASE = ' na základě přiloženého souboru medidetect-test-utf8.csv'
class Medidetect(BaseModel):
    prompt: str
    base: Union[str, None] = None    
    filter: Union[str, None] = None

app = FastAPI()
origins = [
    "https://egolem.online",
    "http://localhost",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import os
#openai.api_key = os.getenv("OPENAI_API_KEY")


@app.get("/")
def read_root():
    return {"AIGolem": "v1.0.0"}


@app.post("/items/")
async def create_medidetect(item: Medidetect):
    aiprompt = ''

    aiprompt = item.prompt;
    
    #if item.filter != None:
    #    aiprompt = 'Vypiš "pacientID" těch pacientů jejichž "popis" svědčí o diagnóze '+item.filter;
    
    
    #if item.base != None:
    #    aiprompt += item.base;
    #else:
    #    aiprompt += AIGOLEMBASE;
    
    completion = client.chat.completions.create(
      model="gpt-4-1106-preview",
        messages=[
        {"role": "system", "content": "You are factual assistant, you follow scientific facts and answers are brief."},
        {"role": "user", "content": aiprompt}
    ])

    response = completion.choices[0].message

    print(response)
    return response.content;

