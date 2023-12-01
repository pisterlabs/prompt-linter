from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
import re
import tiktoken
import time
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

from reddit import Post, Comment, initialize_db
from config import config

pinecone.init(
    api_key=config["pinecone_db"]["api_key"],
    environment=config["pinecone_db"]["environment"]
)


def get_token_count(string: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def object_as_dict(obj):
    return {column.key: getattr(obj, column.key) for column in inspect(obj).mapper.column_attrs}


def comments_to_vectorize():
    session = initialize_db()

    unvectorized_comments = session.query(
        Comment).filter_by(vectorized=False).all()

    comments_as_dict = [object_as_dict(comment)
                        for comment in unvectorized_comments]

    output = []
    for comment_dict in comments_as_dict:
        post = session.query(Post).filter_by(
            id=comment_dict["post_id"]).first()
        if post:
            comment_dict["postData"] = object_as_dict(post)
        else:
            comment_dict["postData"] = None

        output.append(comment_dict)

    return output


def clean_str(content):
    s = content.replace("\n", "")
    s = re.sub(' +', ' ', s)
    s = s.strip()
    return s


# TODO: this will most likely need adjusting
def create_prompt(all_post, comment_content):
    succinct_content = "A human in a forum reads the following headline: {} And responds by saying: {}".format(
        all_post,
        comment_content
    )
    return succinct_content


def initilize_pinecone(index_name, dimension, metric):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=dimension,
            metric=metric
        )

        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(3)


def vectorize(embed_model, model, token_limit, index_name):
    session = initialize_db()
    comments = comments_to_vectorize()

    print("---> Going To Upload: {} comments".format(len(comments)))

    data = []
    for comment in comments:
        if type(comment["postData"]) != dict:
            continue

        post_id = clean_str(comment["post_id"])
        post_title = clean_str(comment["postData"]["title"])
        post_content = clean_str(comment["postData"]["selftext"])
        post_subreddit = clean_str(comment["postData"]["subreddit"])

        comment_id = clean_str(comment["id"])
        comment_content = clean_str(comment["comment"])
        comment_scraped = comment["recorded"]

        all_post = "{}. {}".format(post_title, post_content)

        prompt = create_prompt(all_post, comment_content)
        meta_data = {
            "post": post_id,
            "comment": comment_id,
            "subreddit": post_subreddit,
            "time": comment_scraped
        }

        token_count = get_token_count(prompt, model)

        MIN_TOKEN = 6
        if (token_count <= MIN_TOKEN):
            continue

        while token_count > token_limit:
            if len(all_post.split()) > len(comment_content.split()):
                all_post = " ".join(all_post.split()[:-1])
            else:
                comment_content = " ".join(comment_content.split()[:-1])

            if len(all_post) == 0 or len(comment_content) == 0:
                raise Exception(
                    "can not reduce (all_post + comment_content) due to token limit {} being too small".format(token_limit))

        data.append({
            "emb_content": prompt,
            "meta_data": meta_data
        })

    index = pinecone.Index(index_name)

    batch_size = 100
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        ids = [
            f"{item['meta_data']['post']}-{item['meta_data']['comment']}" for item in batch]
        texts = [item['emb_content'] for item in batch]

        # Embed text
        embeds = embed_model.embed_documents(texts)

        # Get metadata to store
        metadata = [
            {
                'post': item['meta_data']['post'],
                'comment': item['meta_data']['comment'],
                'subreddit': item['meta_data']['subreddit'],
                'time': item['meta_data']['time']
            } for item in batch
        ]

        # Add to Pinecone
        index.upsert(vectors=zip(ids, embeds, metadata))

        # Update the 'vectorized' column for each comment in the batch
        for item in batch:
            cid = item["meta_data"]["comment"]
            session.query(Comment).filter_by(
                id=cid).update({Comment.vectorized: True})

    # Commit the changes to the session
    session.commit()
    session.close()
    print("DONE!")
