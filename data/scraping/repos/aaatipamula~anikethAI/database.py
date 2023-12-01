import json
from os.path import join, dirname
from langchain.memory import ConversationBufferWindowMemory, ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from sqlalchemy import (
    Text,
    Integer,
    select,
    update,
    delete,
    create_engine
)
from sqlalchemy.orm import (
    DeclarativeBase,
    mapped_column,
    Session,
    Mapped
)
db_path = join(dirname(__file__), 'data', 'bot.db')

engine = create_engine(
    "sqlite:///" + db_path
)

MEM_LEN = 7

class BaseModel(DeclarativeBase): ...

class User(BaseModel):
    __tablename__ = "Users"

    id: Mapped[int] = mapped_column(primary_key=True)
    memory: Mapped[str] = mapped_column(Text(), default="{}")

class StarredMessage(BaseModel):
    __tablename__ = "StarredMessages"

    id: Mapped[int] = mapped_column(primary_key=True)
    board_message: Mapped[int] = mapped_column(Integer())

def get_board_message_id(_id: int) -> int:
    stmt = select(StarredMessage.board_message).where(StarredMessage.id==_id)
    with Session(engine) as session:
        result = session.execute(stmt).first()
    return result[0] if result else 0

def add_starred_message(_id: int, board_message: int):
    with Session(engine) as session:
        new_starred_msg = StarredMessage(id=_id, board_message=board_message)
        session.add(new_starred_msg)
        session.commit()

def remove_starred_message(_id: int):
    stmt = delete(StarredMessage).where(StarredMessage.id==_id)
    with Session(engine) as session:
        session.execute(stmt)
        session.commit()

def get_user_mem(_id: int) -> ConversationBufferWindowMemory:
    stmt = select(User.memory).where(User.id == _id)
    history = None
    with Session(engine) as session:
        result = session.execute(stmt).first()
        if result:
            memory_dict = json.loads(result[0])
            messages = messages_from_dict(memory_dict)
            history = ChatMessageHistory(messages=messages)
        else:
            new_user = User(id=_id)
            session.add(new_user)
            session.commit()
    history = history if history else ChatMessageHistory()

    return ConversationBufferWindowMemory(
        chat_memory=history,
        return_messages=True,
        memory_key="history",
        k=MEM_LEN
    )

def dump_user_mem(_id: int, memory: ConversationBufferWindowMemory) -> None:
    memory_dict = messages_to_dict(memory.buffer)
    memory_str = json.dumps(memory_dict)
    stmt = update(User).where(User.id==_id).values(memory=memory_str)
    with Session(engine) as session:
        session.execute(stmt)
        session.commit()

