from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import crud
import schemas
from utils.utils import get_db

import openai
from secrets import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


router = APIRouter(
    tags=["keywords"],
)


async def get_new_keywords(case: schemas.Case):
    prompt = "A lawyer needs to choose keywords to classify the following text: {}\n\nThe lawyer lists the following 5 keywords:".format(
        case.narrative)

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.3,
        max_tokens=30,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0
    )
    return response.choices[0]['text']


@router.post('/cases/{case_id}/keywords', response_model=schemas.Keyword)
async def create_keywords_for_case(case_id: int, db: Session = Depends(get_db)):
    db_case = crud.get_case(db, case_id=case_id)
    if db_case is None:
        raise HTTPException(
            status_code=404, detail="Case not found, post to /keywords to create new case and keywords")

    new_keywords = await(get_new_keywords(db_case))

    return crud.create_keywords(db=db, keyword=new_keywords, case_id=case_id)


@router.post('/keywords', response_model=schemas.Keyword)
async def create_keywords(case: schemas.CaseCreate, db: Session = Depends(get_db)):
    db_case = crud.get_case(db, case_id=case.case_id)
    if db_case is None:
        db_case = crud.create_case(db=db, case=case)

    new_keywords = await(get_new_keywords(db_case))

    return crud.create_keywords(db=db, keyword=new_keywords, case_id=db_case.case_id)
