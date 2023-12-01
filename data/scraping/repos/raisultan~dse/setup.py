import openai
from aiohttp.web import Application
from pymilvus import Collection, connections

from app.schemas.ads import collection_name, schema
from app.collections.ads import AdsCollection

async def connect_milvus(app: Application) -> None:
    connections.connect(
        alias="default",
        user='username',
        password='password',
        host=app['config']['milvus_host'],
        port=app['config']['milvus_port'],
    )


async def disconnect_milvus(_: Application) -> None:
    connections.disconnect("default")


async def setup_openai(app: Application) -> None:
    openai.api_key = app['config']['openai_secret']


async def ad_collection(app: Application) -> None:
    app["ads_collection"] = AdsCollection(
        Collection(name=collection_name, schema=schema),
    )
