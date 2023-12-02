from pydantic import BaseModel
from langchain.prompts import HumanMessagePromptTemplate

class UserTemplate(BaseModel):
    """
    User Template Class
    """
    id: str = "template id"
    name: str = "name of the template"
    content_user_gen: HumanMessagePromptTemplate = "user generation template"
    content_user_con: HumanMessagePromptTemplate =  "task connectives generation template"