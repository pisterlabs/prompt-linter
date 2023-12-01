import json
import re
from dataclasses import dataclass, field

import chromadb
import httpx
import openai
import requests

# RTZR
from src import DB_DIR
from src.common.logger import logger

tistory_access_token = "63bca7765c9d6cdf89555d7c1c933543_fee3bbb1ea9c6f352bcef01f4417663a"
tistory_blog_name = "memoge"
# response_dict, status = await fetch(url, method="POST", headers=headers, params=params, data=request_body)
user_info = {"link": {"category_id": 1172982}, "fran": {"category_id": 1172983}}


@dataclass
class Article:
    id: str
    writer: str  # category-id
    title: str
    content: str
    tags: list[str] = field(default_factory=list)


async def write_post(
    title: str,
    content: str,
    category=0,
    visibility=3,
    published=None,
    slogan=None,
    tag=None,
    acceptComment=1,  # 허용
    password=None,
):
    """https://tistory.github.io/document-tistory-apis/apis/v1/post/write.html"""
    # API endpoint
    url = "https://www.tistory.com/apis/post/write"

    # Parameters
    params = {
        "access_token": tistory_access_token,
        "output": "json",
        "blogName": tistory_blog_name,
        "title": title,
        "visibility": visibility,
        "category": category,
        "acceptComment": acceptComment,
    }

    # Optional parameters
    if content:
        params["content"] = content
    if published:
        params["published"] = published
    if slogan:
        params["slogan"] = slogan
    if tag:
        params["tag"] = tag
    if password:
        params["password"] = password

    # Make the API request
    async with httpx.AsyncClient() as client:
        response = await client.post(url, params=params)
        data = response.json()

    # Check for errors
    if data["tistory"]["status"] != "200":
        print("Error:", data["tistory"]["status"])
        return None

    # Return the post ID and URL
    return data["tistory"]["postId"], data["tistory"]["url"]


async def read_post(post_id: str | int):
    post_id = str(post_id)
    # API endpoint
    url = "https://www.tistory.com/apis/post/read"

    # Parameters
    params = {"access_token": tistory_access_token, "blogName": tistory_blog_name, "postId": post_id, "output": "json"}

    # Make the API request
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)

    # Parse the response
    try:
        data = response.json()
    except requests.JSONDecodeError:
        print("Error: The API response is not in JSON format: \n", response.text)
        return None

    # Check for errors
    if data["tistory"]["status"] != "200":
        print("Error:", data["tistory"]["status"])
        return None

    # Return the post details
    return data["tistory"]["item"]


async def get_post_ids():
    url = "https://www.tistory.com/apis/post/list"
    params = {
        "access_token": tistory_access_token,
        "output": "json",
        "blogName": tistory_blog_name,
        "page": 1,
    }

    posts = []
    while True:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            data = response.json()
        if posts_ := data["tistory"]["item"].get("posts"):
            posts.extend(posts_)
        else:
            break
        params["page"] += 1

    return [post["id"] for post in posts]


async def save_new_posts_to_db():
    post_dir = DB_DIR / "posts"
    if not post_dir.exists():
        post_dir.mkdir()
    post_ids_saved = [int(path.stem) for path in post_dir.glob("*.json")]

    post_ids_tistory = await get_post_ids()
    for post_id in post_ids_tistory:
        if post_id in post_ids_saved:
            continue
        post = await read_post(post_id)
        with open((post_dir / f"{post_id}.json"), "w") as f:
            json.dump(post, f, ensure_ascii=False, indent=4)


def embed_fn(text: str | list[str]):
    resp = openai.Embedding.create(engine="rtzr-embed-ada-cscat", input=text)
    return [embed["embedding"] for embed in resp["data"]]


def construct_vectordb():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection("memoge", embedding_function=embed_fn)
    post_dir = DB_DIR / "posts"
    post_ids_saved = [path.stem for path in post_dir.glob("*.json")]

    for post_id in post_ids_saved:
        if collection.get(post_id)["ids"]:
            continue
        with open((post_dir / f"{post_id}.json"), "r") as f:
            post = json.load(f)
        collection.add(
            ids=post["id"],
            documents=post["content"],
            metadatas={"title": post["title"]},
        )

    logger.info(f"Collection: {collection.count()}")


def add_to_vectordb(post_id, content):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection("memoge", embedding_function=embed_fn)
    collection.add(
        ids=post_id,
        documents=content,
    )


async def upload_comment(comment, post_id):
    url = "https://www.tistory.com/apis/comment/write?"
    params = {
        "access_token": tistory_access_token,
        "blogName": tistory_blog_name,
        "postId": post_id,
        "content": comment,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, params=params)
        assert response.status_code == 200


def remove_html_tags(text):
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def generate_comment(diary, k=2):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection("memoge", embedding_function=embed_fn)
    diary_embed = embed_fn(diary)
    related_diaries = collection.query(query_embeddings=diary_embed, n_results=k)
    comments = []
    for r_diary, rid, r_meta in zip(
        related_diaries["documents"][0], related_diaries["ids"][0], related_diaries["metadatas"][0]
    ):
        r_diary = remove_html_tags(r_diary)
        resp = openai.ChatCompletion.create(
            engine="rtzr-gpt-4-cscat",
            messages=[
                {"role": "user", "content": f"User's diary\n{diary}"},
                {"role": "user", "content": f"Related diary\n{r_diary}"},
                {
                    "role": "user",
                    "content": "Read the user's diary and related one. Write a comment freely from the point of view of related one's author to the user after reading them, in Korean, within 3 sentences.",
                },
            ],
            temperature=0.7,
        )
        comment = resp["choices"][0]["message"]["content"]
        comment += "\n연관 일기:"
        if r_meta and (title := r_meta.get("title")):
            comment += title
        comment += f"\nhttps://memoge.tistory.com/{rid}"
        comments.append(comment)

    return comments
