from fastapi import APIRouter, HTTPException, Depends, Response, Request
from fastapi.responses import ORJSONResponse

from app.core.config import settings
from app.schema.user_schema import UserForm, UserToken, UserLoginForm, ChatQuestionForm, ChatResponseForm
from app.crud.user_crud import create_user, get_user_by_id, verfiy_password, create_access_token, get_current_user, get_user, get_recent_chats
from app.db.model import TChats
from app.db.connection import get_db
from app.api.endpoints.talk import qdrant_settings, build_prompt, save_ask

import openai

from datetime import timedelta, datetime
from typing import Annotated

from sqlalchemy.orm import Session

from qdrant_client import QdrantClient

from jose import jwt

openai.api_key = settings.OPENAI_API_KEY

COLLECTION_NAME = 'anubot-unified'

qdrant_client = QdrantClient(
    url = settings.QDRANT_URL,
    port= settings.QDRANT_PORT, 
)
router = APIRouter()


def chat_ask(question, id, session: Session):
    similar_docs = qdrant_client.search(
        collection_name='anubot-unified',
        query_vector=openai.Embedding.create(input=question, model=settings.EMBEDDING_MODEL)["data"][0]["embedding"],
        limit=5,
        append_payload=True,
    )
    saved_ask = get_recent_chats(id, session)

    print('생성중')
    prompt= build_prompt(question, similar_docs, saved_ask)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.2,
    )

    final_response = response["choices"][0]["message"]["content"]

    save_ask(question, final_response, id, session)
    
    return final_response

@router.post(path="/signup", description="Register form")
async def signup(new_user: UserForm, db: Session = Depends(get_db)):
    user = get_user_by_id(new_user.id, db)

    if user:
        raise HTTPException(status_code=409, detail="User already exists")
    
    await create_user(new_user, db)

    return HTTPException(status_code=200, detail="User created")

@router.post(path="/login", description="Login form")
async def login(response: Response, user: UserLoginForm, db: Session = Depends(get_db)):
    db_user = get_user_by_id(user.id, db)

    if not db_user:
        raise HTTPException(status_code=401, detail="User not found")
    if not verfiy_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Incorrect password")

    access_token_expoires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expoires
    )

    response.set_cookie(key="access_token", value=access_token, expires=access_token_expoires, samesite='none', secure=True)

    recent_chats = get_recent_chats(db_user.id, db)

    return UserToken(access_token=access_token, token_type="bearer", recent_chats=recent_chats, username=db_user.name, id=db_user.id)

@router.get(path="/logout", description="Logout form")
async def logout(response: Response):

    response.delete_cookie(key="access_token")

    return HTTPException(status_code=200, detail="Logout success")

@router.post(path="/webchat", description="Chat with AnuBot")
async def web_chat(item: ChatQuestionForm, request: Request, db: Session = Depends(get_db)):
    if request.cookies.get("access_token") is None:
        raise HTTPException(status_code=401, detail="Please login before chat")

    user = await get_current_user(request.cookies.get("access_token"), db)

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    chat_result = chat_ask(item.chat, user.id, db)
    
    return ORJSONResponse(content={"reply": chat_result})

