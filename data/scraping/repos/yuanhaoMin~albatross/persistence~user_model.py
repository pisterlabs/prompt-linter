from configuration.database import Base

# Although the following import is not used in this file, it is required for the relationship to work.
from persistence.openai_completion_model import OpenAICompletion
from persistence.openai_chat_completion_model import OpenAIChatCompletion

from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.orm import relationship


class AppUser(Base):
    __tablename__ = "app_user"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True)
    password = Column(String)
    access_bitmap = Column(Integer, default=1)
    created_time = Column(TIMESTAMP)
    last_login_time = Column(TIMESTAMP)
    subscription_end_time = Column(TIMESTAMP)
    # One-to-one relationship
    completion = relationship("OpenAICompletion", uselist=False, backref="app_user")
    # One-to-many relationship
    chat_completions = relationship("OpenAIChatCompletion", backref="app_user")
