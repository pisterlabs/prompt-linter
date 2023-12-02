
import asyncio
import uuid
from typing import List, Union

from aredis_om.model import NotFoundError
from fastapi import APIRouter, File, Form, UploadFile, status
from pydantic import UUID4, BaseModel, Field, HttpUrl
from starlette.responses import Response
from typing_extensions import Annotated

from app.exceptions import RoomChainNotFoundException, RoomNotFoundException
from app.services import room_service
from app.services.room_service import RoomInputType
from infra import llm, loader
from infra.redis import Room

router = APIRouter(prefix='/rooms')

class CreateRoomUrlRequest(BaseModel):
    # TODO:
    # partner_uuid: UUID4
    title: str = Field(title="챗봇명")

    urls: List[HttpUrl] = Field(title="챗봇이 학습할 사이트들", description="POST /pages에서 url들을 추출하여 전달하면 된다.")
    
    prompt: Union[str, None] = Field(
        default=llm.DEFAULT_PROMPT_TEMPLATE, 
        title="챗봇이 QA할 prompt",
    )


class RoomResponse(BaseModel):
    room_uuid: UUID4


@router.get("/{pk}", status_code=200)
async def get_a_room(pk: str, response: Response):
    try:
        return await Room.get(pk)
    except NotFoundError:
        raise RoomNotFoundException(room_pk=pk)




@router.post(
        "/url", 
        status_code=status.HTTP_202_ACCEPTED,
        response_model=RoomResponse,
        )
async def create_a_room_with_url(room_in: CreateRoomUrlRequest,):
    room_uuid = uuid.uuid4()
    await asyncio.gather(
        room_service.create_a_room(
            room_uuid,
            room_in.title,
            room_in.prompt,
        ),
        room_service.create_a_room_chain(
            room_uuid,
            RoomInputType.URL,
            room_in.urls
        )
    )
    return RoomResponse(room_uuid=room_uuid)



class CreateRoomTextRequest(BaseModel):
    title: str = Field(title="챗봇명")
    texts: str = Field(title="챗봇에게 더 학습시킬 정보")
    prompt: Union[str, None] = Field(
        default=llm.DEFAULT_PROMPT_TEMPLATE, 
        title="챗봇이 QA할 prompt",
    )


@router.post(
        "/text", 
        status_code=status.HTTP_202_ACCEPTED,
        response_model=RoomResponse,
        )
async def create_a_room_with_text(room_in: CreateRoomTextRequest,):
    room_uuid = uuid.uuid4()
    await asyncio.gather(
        room_service.create_a_room(
            room_uuid,
            room_in.title,
            room_in.prompt,
        ),
        room_service.create_a_room_chain(
            room_uuid,
            RoomInputType.TEXT,
            room_in.texts
        )
    )
    return RoomResponse(room_uuid=room_uuid)





@router.post(
        "/file", 
        status_code=status.HTTP_200_OK,
        response_model=RoomResponse,
        )
async def create_a_room_with_file(
        title: Annotated[str, Form(description="챗봇명")], 
        prompt: Annotated[str, Form(description="챗봇이 QA할 prompt")]=llm.DEFAULT_PROMPT_TEMPLATE,
        file: UploadFile=File(...),
    ):
    room_uuid = uuid.uuid4()

    (saved_room_uuid, docs) = await asyncio.gather(
        room_service.create_a_room(
            room_uuid,
            title,
            prompt,
        ),
        loader.from_file(file.filename, file.file)
    )

    await room_service.create_a_room_chain(
            room_uuid,
            RoomInputType.FILE,
            data=docs,
    )

    return RoomResponse(room_uuid=saved_room_uuid)

from langchain.docstore.document import Document
from langchain.document_loaders import YoutubeLoader


class YoutubeVideoLoader(YoutubeLoader):
    def load(self):
        """Load documents."""
        try:
            from youtube_transcript_api import (NoTranscriptFound,
                                                TranscriptsDisabled,
                                                YouTubeTranscriptApi)
        except ImportError:
            raise ImportError(
                "Could not import youtube_transcript_api python package. "
                "Please install it with `pip install youtube-transcript-api`."
            )

        metadata = {"source": self.video_id}

        if self.add_video_info:
            video_info = self._get_video_info()
            metadata.update(video_info)

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
        except TranscriptsDisabled:
            ...

        return None
        try:
            transcript = transcript_list.find_transcript([self.language])
        except NoTranscriptFound:
            en_transcript = transcript_list.find_transcript(["en"])
            transcript = en_transcript.translate(self.language)

        transcript_pieces = transcript.fetch()

        transcript = " ".join([t["text"].strip(" ") for t in transcript_pieces])

        return [Document(page_content=transcript, metadata=metadata)]


class CreateRoomYoutubeRequest(BaseModel):
    # TODO:
    # partner_uuid: UUID4
    title: str = Field(title="챗봇명")

    youtube_url: HttpUrl = Field(title="Youtube video URL")
    
    prompt: Union[str, None] = Field(
        default=llm.DEFAULT_PROMPT_TEMPLATE, 
        title="챗봇이 QA할 prompt",
    )
@router.post(
        "/youtube", 
        status_code=status.HTTP_202_ACCEPTED,
        response_model=RoomResponse,
        )
async def create_a_room_with_youtube_video_url(room_in: CreateRoomYoutubeRequest,):
    loader = YoutubeVideoLoader.from_youtube_url(room_in.youtube_url, add_video_info=True)

    docs = loader.load()


    # room_uuid = uuid.uuid4()
    # await asyncio.gather(
    #     room_service.create_a_room(
    #         room_uuid,
    #         room_in.title,
    #         room_in.prompt,
    #     ),
    #     room_service.create_a_room_chain(
    #         room_uuid,
    #         RoomInputType.URL,
    #         room_in.urls
    #     )
    # )
    return RoomResponse(room_uuid="a2a26e0c-0e88-4204-b6d6-3b0a5f5b9557")



class PatchChatRoomByTextRequest(BaseModel):
    title: Union[str, None] = Field(title="챗봇명", default=None)
    texts: str = Field(title="챗봇에게 더 학습시킬 정보")
    prompt: Union[str, None] = Field(title="챗봇이 QA할 prompt", default=None)


@router.patch(
        "/text/{room_uuid}", 
        status_code=status.HTTP_202_ACCEPTED,
        response_model=RoomResponse,)
async def change_a_room_by_text(room_uuid: str, chat_in: PatchChatRoomByTextRequest):
    patch_room_data = chat_in.dict(exclude_unset=True)
    patch_room_data.pop('texts')
    await asyncio.gather(
        room_service.update_a_room(room_uuid, patch_room_data),
        room_service.change_a_room_chain(room_uuid, RoomInputType.TEXT, chat_in.texts)
    )
    return RoomResponse(room_uuid=room_uuid)


class PatchChatRoomByUrlsRequest(BaseModel):
    title: Union[str, None] = Field(title="챗봇명", default=None)
    urls: List[HttpUrl] = Field(title="챗봇이 학습할 사이트들", description="POST /pages에서 url들을 추출하여 전달하면 된다.")
    prompt: Union[str, None] = Field(title="챗봇이 QA할 prompt", default=None)
    


@router.patch(
        "/url/{room_uuid}", 
        status_code=status.HTTP_202_ACCEPTED,
        response_model=RoomResponse,)
async def change_a_room_by_urls(room_uuid: str, chat_in: PatchChatRoomByUrlsRequest):
    patch_room_data = chat_in.dict(exclude_unset=True)
    patch_room_data.pop('urls')
    await asyncio.gather(
        room_service.update_a_room(room_uuid, patch_room_data),
        room_service.change_a_room_chain(room_uuid, RoomInputType.URL, chat_in.urls)
    )
    return RoomResponse(room_uuid=room_uuid)




class AppendChatRoomByTextRequest(BaseModel):
    texts: str = Field(title="챗봇에게 더 학습시킬 정보")


@router.patch(
        "/text/add/{room_uuid}", 
        status_code=status.HTTP_202_ACCEPTED,
        response_model=RoomResponse,)
async def append_a_chat_room_by_text(room_uuid: str, chat_in: AppendChatRoomByTextRequest):
    room_uuid = await room_service.append_a_room_chain(
        room_uuid, 
        RoomInputType.TEXT, 
        chat_in.texts
    )
    return RoomResponse(room_uuid=room_uuid)


class AppendChatRoomByUrlsRequest(BaseModel):
    urls: List[HttpUrl] = Field(title="챗봇이 추가할 사이트들")


@router.patch(
        "/url/add/{room_uuid}", 
        status_code=status.HTTP_202_ACCEPTED,
        response_model=RoomResponse,)
async def append_a_chat_room_by_url(room_uuid: str, chat_in: AppendChatRoomByUrlsRequest):
    room_uuid = await room_service.append_a_room_chain(
        room_uuid, 
        RoomInputType.URL, 
        chat_in.urls
    )
    return RoomResponse(room_uuid=room_uuid)


@router.delete(
        "/{room_uuid}", 
        status_code=status.HTTP_204_NO_CONTENT)
async def delete_a_room(room_uuid: str,):
    (num_of_room_deleted,is_chain_deleted) = await asyncio.gather(
        room_service.delete_a_room(room_uuid),
        room_service.delete_a_room_chain(room_uuid)
    )
    
    if num_of_room_deleted < 1:
        raise RoomNotFoundException(room_pk=room_uuid)

    if is_chain_deleted is False:
        raise RoomChainNotFoundException(room_pk=room_uuid)
