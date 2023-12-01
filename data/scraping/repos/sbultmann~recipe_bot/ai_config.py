from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chat_models import ChatOpenAI

from typing import List

from pydantic import BaseModel, Field

from langchain.chains.openai_functions import create_qa_with_structure_chain
from langchain.chains.openai_functions import create_structured_output_chain

from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

import chromadb
import openai
from app import app

model_name = "intfloat/multilingual-e5-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
llm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

gpt4 = ChatOpenAI(temperature=1, model="gpt-4-1106-preview")#, openai_api_key=openai.api_key)



client = chromadb.HttpClient(host=app.config['CHROMA_HOST_NAME'], port=8000)
collection_name = "GuteKuecheAT"
index = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=llm
    )

class Zutat(BaseModel):
    name: str = Field(..., description="Name der Zutat")
    unit: str = Field(..., description="Maßeinheit der Zutat")
    amount: float = Field(..., description="Menge der Zutat")

class Nutrition(BaseModel):
    name: str = Field(..., description="Art des Nähwerts: Kalorien, Eiweiß, Fett und Kohlehydrate")
    unit: str = Field(..., description="Maßeinheit der Nähwerts")
    amount: float = Field(..., description="Menge der Nähwerts")

class Rezept(BaseModel):
    """Ein leckeres und gesundes Rezept"""
    title: str = Field(..., description="Kreativer Titel für das Rezept")
    prompt: str = Field(..., description="Eine prompt für bildgenerierende AI DALL-E3 ein qualitativ hochwertiges Bild des Rezepts im Stil der modernen Food-Fotografie zu erstellen, 15mm, warmes Licht")
    beschreibung: str = Field(..., description="Answer to the question that was asked")
    portionen: int = Field(..., description="Anzahl der Portionen des Rezepts")
    recipe_type: str = Field(..., description="Angabe ob das Rezept vegan, vegetarisch oder fleisch ist")
    tipp: str = Field(..., description="Ein Tipp zur hilfreich zu Zubereitung oder möglichen Verfeinerung des Rezepts")
    ingredients: List[Zutat] = Field(..., description="Alle Zutaten nötig für die Herstellung des Rezepts")
    instructions: List[str] = Field(..., description="Arbeitsschritte notwendig zu Herstellung des Rezepts. Nicht nummeriert!")
    naehrwerte: List[Nutrition] = Field(..., description="Alle Nähwerte des Rezepts: Kalorien, Eiweiß, Fett und Kohlehydrate")
   

doc_prompt = PromptTemplate(
    template="Rezept: {page_content}\Bewertung: {Bewertung}",
    input_variables=["page_content", "Bewertung"],
)
#prompt for website and text extraction
we_prompt_messages = [
    SystemMessage(
        content=(
            "Du bist der weltbeste algorithmus zum extrahieren von Rezepten aus texten. "
            "Nutzer geben dir texte und die extrahierst daraus das in den Texten vorhandene Rezept"
            "Füge fehlende Information hinzu, um die Formatvorgaben zu erfüllen"
            "Du gibst deine Antwort in einer spezifischen Form zurück."
        )
    ),
    HumanMessage(content="Erstelle das Rezept auf der Basis dieses Textes:"),
    HumanMessagePromptTemplate.from_template("Text: {text}"),
    HumanMessage(
        content="Tips: Stelle sicher, dass du im korrekten Format antwortest!"
    ),
]

we_chain_prompt = ChatPromptTemplate(messages=we_prompt_messages)

we_chain = create_structured_output_chain(Rezept, gpt4, we_chain_prompt)


# prompt for RAG recipe generation
prompt_messages = [
    SystemMessage(
        content=(
            "Du bist der weltbeste algorithmus zum erstellen gesunder und schmackhafter Rezepte. "
            "Nutzer geben die Zutaten und du entwirfst daraus an Hand des dir gegeben Kontext ein neues, gesundes und schmackhaftes Gericht."
            "Füge, wenn notwendig Zutaten hinzu, um das Gericht zu perfektionieren."
            "Du gibst deine Antwort in einer spezifischen Form zurück."
        )
    ),
    HumanMessage(content="Erstelle das Rezept auf der Basis dieser Datenbankeinträge"),
    HumanMessagePromptTemplate.from_template("{context}"),
    HumanMessagePromptTemplate.from_template("Zutaten: {question}"),
    HumanMessage(
        content="Tips: Stelle sicher, dass du im korrekten Format antwortest!"
    ),
]

chain_prompt = ChatPromptTemplate(messages=prompt_messages)

qa_chain_pydantic = create_qa_with_structure_chain(
    gpt4, Rezept, output_parser="pydantic", prompt=chain_prompt
)


final_qa_chain_pydantic = StuffDocumentsChain(
    llm_chain=qa_chain_pydantic,
    document_variable_name="context",
    document_prompt=doc_prompt,
)

retrieval_qa = RetrievalQA(
    retriever=index.as_retriever(search_kwargs={'k': 10, 
                                                "filter":{'$and': [{'Bewertung':{'$gte':4.5}}, {'Stimmen':{'$gte':200}}]}}), 
                                 combine_documents_chain=final_qa_chain_pydantic,
                                 return_source_documents=True,
)

                    
