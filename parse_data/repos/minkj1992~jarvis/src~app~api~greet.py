from datetime import datetime

from fastapi import APIRouter, Request, status
from fastapi.templating import Jinja2Templates

from infra.config import get_config

router = APIRouter()
cfg = get_config()


templates = Jinja2Templates(directory="templates")


@router.get("/")
async def greet(request: Request):
    current_time = datetime.utcnow()
    return templates.TemplateResponse("prompt.html", {"request": request})


@router.get("/ping", status_code=status.HTTP_200_OK)
async def pong():
    return {"ping": "pong!"}
