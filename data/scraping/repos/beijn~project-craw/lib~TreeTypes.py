from pydantic import BaseModel, Field

from langchain.docstore.document import Document



class Question(BaseModel):
    question : str
    failed_alternatives : list = []



class UnanswerableQuestion(BaseModel):
    question : Question
    answer   : str

class Answer(BaseModel):
    answer  : str
    source  : Document | str


class AnsweredQuestion(BaseModel):
    question : Question
    answer   : Answer
    sub_aqs  : list = []
