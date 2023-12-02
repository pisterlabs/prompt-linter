from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from langchain.memory import PostgresChatMessageHistory
from llm_project.database.config import settings
from llm_project.services.auth import auth_service

from llm_project.database.models import MessageHistory, User
from llm_project.database.schemas import HistoryResponse
from llm_project.repository import history as repository_history
from llm_project.database.db import get_db
from typing import List
from sqlalchemy import and_


router = APIRouter(prefix="/history", tags=["history"])


@router.post("/save_massegas")
async def save_messages(text: str, chat_id: str, user_id: str, db: Session = Depends(get_db)):
    new_message = await repository_history.create_message(chat_id, user_id, text, db)
    return new_message


@router.get("/chat_id", response_model= List[HistoryResponse])
async def get_history_messages(chat_id: str, db: Session = Depends(get_db)):
    chat_history = await repository_history.chat_history(chat_id, db)
    if chat_history is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='Chat not found!')
    return chat_history
    
@router.get("/user_id", response_model= List[HistoryResponse])
async def get_history_messages(user_id: str, db: Session = Depends(get_db)):
    chat_history = await repository_history.user_history(user_id, db)
    if chat_history is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='Chat not found!')
    return chat_history   

@router.delete("/message_id")
async def get_history_messages(chat_id:str, message_id: str, db: Session = Depends(get_db)):
    delete_message = db.query(MessageHistory).filter(and_(MessageHistory.chat_id == chat_id, MessageHistory.id == message_id)).first()
    if delete_message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Message not found!')
    db.delete(delete_message)
    db.commit()