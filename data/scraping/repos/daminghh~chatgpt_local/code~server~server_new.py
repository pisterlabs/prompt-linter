from flask import Flask, render_template, request
from qdrant_client import QdrantClient
from itertools import combinations
import openai
import os
import logging

app = Flask(__name__)

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up environment variables
OPENAI_API_KEY = ""

# Initialize the Qdrant client and the OpenAI model
client = QdrantClient("localhost", port=6333)
collection_name = "data_collection"
openai.api_key = OPENAI_API_KEY
similarity_model = openai.Model("text-similarity-002")


def prompt(question, answers):
    demo_q = '使用以下段落来回答问题："成人头疼，流鼻涕是感冒还是过敏？"\n1. 普通感冒：您会出现喉咙发痒或喉咙痛，流鼻涕，流清澈的稀鼻涕（液体），有时轻度发热。\n2. 常年过敏：症状包括鼻塞或流鼻涕，鼻、口或喉咙发痒，眼睛流泪、发红、发痒、肿胀，打喷嚏。'
    demo_a = '成人出现头痛和流鼻涕的症状，可能是由于普通感冒或常年过敏引起的。如果病人出现咽喉痛和咳嗽，感冒的可能性比较大；而如果出现口、喉咙发痒、眼睛肿胀等症状，常年过敏的可能性比较大。'
    system = '你是一个医院问诊机器人'
    q = '使用以下段落来回答问题，如果段落内容不相关就返回未查到相关信息："'
    q += question + '"'
    # 带有索引的格式
    for index, answer in enumerate(answers):
        q += str(index + 1) + '. ' + str(answer['title']) + ': ' + str(answer['text']) + '\n'
    res = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': demo_q},
        {'role': 'assistant', 'content': demo_a},
        {'role': 'user', 'content': q},
    ]
    return res


def query(text):
    try:
        # Use openai to calculate the similarity of the search results to the text
        sentence_embeddings = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        search_result = client.search(
            collection_name=collection_name,
            query_vector=sentence_embeddings["data"][0]["embedding"],
            limit=3,
            search_params={"exact": False, "hnsw_ef": 128}
        )
        answers = [result.payload for result in search_result]

        max_similarity = 0
        best_answer = ""
        for a in answers:
            # Use OpenAI to calculate similarity between search results and text
            score = similarity_model.predict(question=text, model="text-similarity-002", examples=[[a['text'], text]])['results'][0]['output']
            logger.info(f"Score of answer {a}: {score}")
            if score > max_similarity:
                max_similarity = score
                best_answer = a['text']

        # Return the best answer if the similarity is high enough
        if max_similarity > 0.8:
            return {
                "answer": best_answer,
                "tags": []
            }
        else:
            # Use OpenAI to generate a response if the similarity is not high enough
            completion = openai.Completion.create(
                engine="gpt-3.5-turbo",
                prompt=text,
                max_tokens=4096,
                n=1,
                stop=None,
                temperature=0.5,
            )

            return {
                "answer": completion.choices[0].text,
                "tags": []
            }

    except Exception as e:
        logger.error(f"Error querying: {e}")
        return {
            "answer": "Error",
            "tags": []
        }


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        search = data['search']
        res = query(search)

        return {
            "code": 200,
            "data": {
                "search": search,
                "answer": res["answer"],
                "tags": res["tags"],
            },
        }

    except Exception as e:
        logger.error(f"Error searching: {e}")
        return {
            "code": 500,
            "data": {
                "search": "",
                "answer": "Error",
                "tags": []
            }
        }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)