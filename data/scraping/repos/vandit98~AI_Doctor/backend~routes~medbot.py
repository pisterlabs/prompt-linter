from fastapi import APIRouter, Query

from dotenv import load_dotenv
import uvicorn

import os
from langchain.llms import OpenAI
from bardapi import Bard
import os
os.environ['_BARD_API_KEY']="WQg7s2cdCPSJ-l4dF2YJogD5wVi4JcQbr9Vi_mS0nCBpjKQfN3jYPUNWzm74KT7JiJLhsQ."

medbot_api_router = APIRouter()

# os.environ['OPENAI_API_KEY'] = "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"
# llm = OpenAI(temperature=0)
# prefix="answer considering yourself as a doctor and if it is not relevant to doctor background say that it is not a relevant question for a doctor"
# suffix="Summarise your answer"


@medbot_api_router.get("/bot")
def medbot(text:str):
    
    # return {'medboat':llm(prefix+text+suffix)}
    response=Bard().get_answer(text)['content']
    # return {'medbot': response}
    return response