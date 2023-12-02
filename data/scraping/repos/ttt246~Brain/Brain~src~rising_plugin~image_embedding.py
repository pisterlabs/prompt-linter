from langchain.embeddings.openai import OpenAIEmbeddings

from ..common.utils import OPENAI_API_KEY, PINECONE_NAMESPACE, PINECONE_INDEX_NAME
from .pinecone_engine import (
    init_pinecone,
    get_pinecone_index_namespace,
    delete_pinecone,
    update_pinecone,
)
from ..model.basic_model import DataStatus
from ..model.image_model import ImageModel
from ..model.req_model import ReqModel


def get_embeddings(setting: ReqModel):
    return OpenAIEmbeddings(openai_api_key=setting.openai_key)


# def embed_image_text(image_text: str, image_name: str, uuid: str) -> str:
def embed_image_text(image: ImageModel, setting: ReqModel) -> str:
    prompt_template = f"""
        This is the text about the image.
        ###
        {image.image_text}
        """

    embed_image = get_embeddings(setting=setting).embed_query(prompt_template)
    index = init_pinecone(index_name=PINECONE_INDEX_NAME, setting=setting)

    """create | update | delete in pinecone"""
    pinecone_namespace = get_pinecone_index_namespace(image.uuid)
    try:
        if image.status == DataStatus.CREATED:
            """add a data in pinecone"""
            upsert_response = index.upsert(
                vectors=[{"id": image.image_name, "values": embed_image}],
                namespace=pinecone_namespace,
            )
        elif image.status == DataStatus.DELETED:
            delete_pinecone(namespace=pinecone_namespace, key=image.image_name)
        elif image.status == DataStatus.UPDATED:
            update_pinecone(
                namespace=pinecone_namespace, key=image.image_name, value=embed_image
            )
    except Exception as e:
        return "fail to embed image text"
    return "success to embed image text"


def query_image_text(image_content, message, setting: ReqModel):
    embed_image = get_embeddings(setting=setting).embed_query(
        get_prompt_image_with_message(image_content, message)
    )
    index = init_pinecone(index_name=PINECONE_INDEX_NAME, setting=setting)
    relatedness_data = index.query(
        vector=embed_image,
        top_k=3,
        include_values=False,
        namespace=get_pinecone_index_namespace(setting.uuid),
    )
    if len(relatedness_data["matches"]) > 0:
        return relatedness_data["matches"][0]["id"]
    return ""


def get_prompt_image_with_message(image_content, message):
    prompt_template = f"""
                This is the text about the image.
                ###
                {image_content}
                ###
                This message is the detailed description of the image.
                ### 
                {message}
                """

    return prompt_template
