from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.src.langchain.service import LangchainService
from backend.src.langchain.model import LangchainRunDTO
from backend.config import Settings, get_settings

from langchain.document_loaders import WebBaseLoader, PyPDFLoader

router = APIRouter()

@router.post('/run')
async def run(param: LangchainRunDTO, settings: Settings = Depends(get_settings)):
    langchainService = LangchainService(settings)

    result = langchainService.runGPT(param.prompt)
    return {'message': result}

@router.post('/upload')
async def upload(
    type: Annotated[str, Form()],
    youtube_link: Annotated[str, Form()] = None,
    web_page: Annotated[str, Form()] = None,
    file: Annotated[UploadFile, File()] = None
):
    return {
        'type': type,
        'youtube_link': youtube_link,
        'web_page': web_page,
        'file-type': file.content_type if file is not None else "No file uploaded"
    }