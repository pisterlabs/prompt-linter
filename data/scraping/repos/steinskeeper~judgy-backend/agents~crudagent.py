from bson import ObjectId
from fastapi import APIRouter, Request
from pymongo import MongoClient
from agents.codeagent import invoke_code_agent
import asyncio
from agents.marketagent import invoke_market_agent
from annoy import AnnoyIndex
import cohere
import numpy as np
import pandas as pd
import os
router = APIRouter()
client = MongoClient("mongodb://localhost:27017/")
db = client["judgy"]


@router.get("/crud-agent")
def crudAgent_endpoint():
    return {"message": "Hello from Crud Agent, Okay I'm not really an agent"}


@router.post("/create-project")
async def create_project(request: Request):
    data = await request.json()
    print(data)
    data["isReviewed"] = False
    new_project = db.projects.insert_one(data)
    print(new_project.inserted_id)

    asyncio.create_task(invoke_market_agent(
        str(new_project.inserted_id), data["shortDescription"]))
    asyncio.create_task(invoke_code_agent(
        data["githubLink"], str(new_project.inserted_id)))
    return {"message": "Project created", "project_id": str(new_project.inserted_id)}


@router.post("/create-hackathon")
async def create_hackathon(request: Request):
    data = await request.json()
    print(data)

    new_hackathon = db.hackathons.insert_one(data)
    print(new_hackathon.inserted_id)

    return {"message": "Hackathon created", "hackathon_id": str(new_hackathon.inserted_id)}


@router.get("/get-project/{project_id}")
async def get_project(project_id: str):
    project = db.projects.find_one({"_id": ObjectId(project_id)})
    project["_id"] = str(project["_id"])
    return {"message": "successful", "project": project}


@router.get("/get-all")
async def get_all_projects():
    projects = db.projects.find({})
    final = []
    for project in projects:
        project["_id"] = str(project["_id"])
        final.append(project)
    return {"message": "successful", "projects": final.reverse()}


@router.post("/review")
async def review_project(request: Request):
    data = await request.json()
    project_id = data["project_id"]
    query = {"_id": ObjectId(project_id)}
    new_values = {"$set": {"isReviewed": data["isReviewed"]}}
    db.projects.update_one(query, new_values)
    return {"message": "successful", "project_id": project_id}


@router.post("/search")
async def search_projects(request: Request):
    data = await request.json()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(cohere_api_key)
    projects = []
    for x in db.projects.find({}):
        x["_id"] = str(x["_id"])
        projects.append(x)
    docs = []
    for y in projects:
        text = "Project Description: " + \
            y["longDescription"] + " Hackathon Theme: " + y["theme"]
        docs.append(text)

    df = pd.DataFrame({'docs': docs})
    embeds = co.embed(texts=list(df["docs"]),
                      model="large",
                      truncate="RIGHT").embeddings
    embeds = np.array(embeds)
    search_index = AnnoyIndex(embeds.shape[1], 'angular')
    for i in range(len(embeds)):
        search_index.add_item(i, embeds[i])
    search_index.build(10)

    query_embed = co.embed(texts=[data["query"]],
                           model="large",
                           truncate="RIGHT").embeddings
    similar_item_ids = search_index.get_nns_by_vector(
        query_embed[0], 10, include_distances=True)
    result = similar_item_ids[0]
    ress = []
    for i in range(len(result)):
        ress.append(projects[result[i]])
    return {"message": "successful", "projects": ress}

