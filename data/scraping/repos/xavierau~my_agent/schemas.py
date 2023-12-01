import uuid
from array import array
from datetime import datetime
from typing import Optional, List

import openai
from pydantic import BaseModel


class TooCallFunction(BaseModel):
    arguments: str
    name: str


class TooCall(BaseModel):
    id: str
    function: TooCallFunction
    type: str


class AgentBase(BaseModel):
    name: str
    configuration: dict | None
    type: str


class AgentCreate(AgentBase):
    pass


class Agent(AgentBase):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class MessageBase(BaseModel):
    role: str
    message: dict


class MessageCreate(MessageBase):
    session_id: uuid.UUID


class Message(MessageBase):
    id: uuid.UUID
    session_id: uuid.UUID

    class Config:
        orm_mode = True


class EntityBase(BaseModel):
    key: str
    content: str | dict


class EntityCreate(EntityBase):
    session_id: uuid.UUID


class EntityEdit(EntityBase):
    pass


class Entity(EntityBase):
    id: uuid.UUID
    session_id: uuid.UUID

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    name: str
    email: str


class UserCreate(UserBase):
    password: str
    phone: Optional[str] = None


class UserEdit(UserBase):
    id: uuid.UUID
    password: str
    is_active: bool


class User(UserBase):
    id: uuid.UUID
    is_active: bool

    class Config:
        orm_mode = True


class SessionBase(BaseModel):
    title: str
    content: str


class SessionCreate(SessionBase):
    user_id: uuid.UUID
    agent_id: uuid.UUID


class Session(SessionBase):
    id: uuid.UUID
    user_id: uuid.UUID
    agent_id: uuid.UUID

    class Config:
        orm_mode = True


# Tools
class NoteBase(BaseModel):
    title: str
    content: str


class NoteCreate(NoteBase):
    user_id: uuid.UUID
    session_id: uuid.UUID | None


class Note(NoteBase):
    id: uuid.UUID
    user_id: uuid.UUID
    session_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
