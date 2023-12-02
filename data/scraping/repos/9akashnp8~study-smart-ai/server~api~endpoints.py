import os
import tempfile
from queue import Queue
from fastapi import (
    UploadFile,
    APIRouter
)
from fastapi.responses import (
    Response,
    JSONResponse
)
from decouple import config
from langchain.llms.openai import OpenAI
from storage3.utils import StorageException
from sse_starlette.sse import EventSourceResponse
from langchain.embeddings.openai import OpenAIEmbeddings

from utils.db import supabase
from core.prompts import (
    flash_card_prompt_template,
)
from core.functions import (
    query_db,
    pdf_to_text_chunks,
    create_embeddings
)
from .models import QueryModel, UserMessage
from core.streaming_chain import StreamingConversationChain
from utils.functions import clean_flashcard_response
from utils.constants import mock_flashcard_response

router = APIRouter()
message_queue = Queue()
llm = OpenAI(openai_api_key=config('OPENAI_API_KEY'), temperature=1)
embedding = OpenAIEmbeddings(openai_api_key=config('OPENAI_API_KEY'))
streaming_response_chain = StreamingConversationChain(
    openai_api_key=config('OPENAI_API_KEY'),
    temparature=1
)

@router.post("/uploadfile/")
async def upload_file(payload: UploadFile):
    """Store file temporarily to create
    embeddings (local) & upload file to cloud
    once done.

    Args:
        payload (UploadFile): the pdf document to create
        embeddings for
    """
    if payload.content_type != 'application/pdf':
        return {'message': 'Please upload a PDF file'}

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        file_path = os.path.join(temp_dir, payload.filename)

        with open(file_path, "wb") as f:
            data = await payload.read()
            f.write(data)

        chunks = pdf_to_text_chunks(file_path=file_path)
        create_embeddings(docs=chunks, collection_name=payload.filename)
        res = supabase.storage.from_('document').upload(
            payload.filename,data, {"content-type": "application/pdf"})

    return {"message": "embeddings created successfully"}

@router.get("/getfile/")
def get_file(file_name: str):
    """get file from cloud, send file
    bytes directly as response

    Args:
        file_name (str): name of file given
    when uploading to cloud. Will come from documents
    list page as query param.

    TODO: store file locally? downloading file everytime
    not efficient.
    """
    try:
        res = supabase.storage.from_('document').download(file_name)
    except StorageException:
        return JSONResponse({"message": "Requested File Not Found"}, status_code=404)
    else:
        return Response(content=res, media_type="application/pdf")

@router.get("/allfiles/")
def get_all_files():
    """List all files from cloud"""
    res = supabase.storage.from_('document').list()
    res.pop(0) # remove default 'emptyFolderPlace' item, see Issue #9155 on supabase.
    return res

@router.post('/flashcard/')
def generate_flashcard(payload: QueryModel, fileName: str, number: int = 1, mock: bool = False):
    """Generates "number" number of flashcards for a given topic
    by the user

    Args:
        payload (QueryModel): a json with just a query key
            having the actual user topic as it's value
        number (int, optional): number of flashcards to generate.
            Defaults to 1.

    Returns:
        a json with the response key having the response
        from OpenAI
    """
    if mock:
        return {"response": mock_flashcard_response}
    context = query_db(payload.query, fileName)
    prompt = flash_card_prompt_template.format(
        number=number,
        context=' '.join(context), topic=payload.query
    )
    response = clean_flashcard_response(llm(prompt))
    return {"response": response}

@router.post('/stream')
async def add_msg_to_queue(user_message: UserMessage):
    """stream endpoint to chat with gen ai.

    Args:
        user_message (UserMessage): Prefix user_message with "/doc"
    to chat with document, without to chat in general (but with context
    of previous chat)
    """
    message_queue.put(user_message)
    return JSONResponse({"status": "success"})


@router.get('/stream', response_class=EventSourceResponse)
async def message_stream():
    """stream endpoint to chat with gen ai.

    Args:
        user_message (UserMessage): Prefix user_message with "/doc"
    to chat with document, without to chat in general (but with context
    of previous chat)
    """
    if not message_queue.empty():
        user_message: UserMessage = message_queue.get()
        return EventSourceResponse(
            streaming_response_chain
            .generate_response(user_message.message, "s3-gsg.pdf")
        )
    return EventSourceResponse(content=iter(()))
