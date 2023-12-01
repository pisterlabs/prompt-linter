import pandas as pd
import json
import openai


with open('server/model/utils/config.json') as config_file:
    config_data = json.load(config_file)
    openai.api_key = config_data["chatgpt"]["secret"]


def get_embedding(text):
    result = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=text)
    return result["data"][0]["embedding"]


def get_query_embed():
    with open('data/augmented/final.json', 'r', encoding='utf-8') as f:
        embeddings = json.load(f)

    df = {}
    index = 0
    for k, value in embeddings.items():
        for query in embeddings[k]['questions']:
            index += 1
            if (index < 1590):
                continue
            embed = get_embedding(query)
            row = {
                'index': value['index'],
                'query': query,
                'embedding': embed
            }
            df[index] = row

            with open('preprocess/embedding/query_embed_back.json', 'w', encoding='utf-8') as f:
                json.dump(df, f, ensure_ascii=False, indent=2)

        print(f"[{index} | {462 * 5}] ")


def merge_query_embed():
    with open('preprocess/embedding/query_embed.json', 'r', encoding='utf-8') as f:
        query_embed = json.load(f)

    with open('preprocess/embedding/query_embed_back.json', 'r', encoding='utf-8') as f:
        query_embed_back = json.load(f)

    for k, value in query_embed_back.items():
        query_embed[k] = value

    with open('preprocess/embedding/query_embed.json', 'w', encoding='utf-8') as f:
        json.dump(query_embed, f, ensure_ascii=False, indent=2)


def test_embed_keys_and_nums():
    with open('preprocess/embedding/query_embed.json', 'r', encoding='utf-8') as f:
        query_embed = json.load(f)

    print(len(query_embed.keys()))
    print(len(query_embed['1']['embedding']))


def check_duplicated_query():
    with open('preprocess/embedding/query_embed.json', 'r', encoding='utf-8') as f:
        query_embed = json.load(f)

    for k, value in query_embed.items():
        for kk, vv in query_embed.items():
            if k == kk:
                continue
            if value['query'] == vv['query']:
                print(k, kk, value['query'], vv['query'])


def check_each_index_has_five_query():
    with open('preprocess/embedding/query_embed.json', 'r', encoding='utf-8') as f:
        query_embed = json.load(f)

    query_count = {}
    for k, value in query_embed.items():
        if value['index'] not in query_count.keys():
            query_count[value['index']] = 0
        query_count[value['index']] += 1

    for k, value in query_count.items():
        if value != 5:
            print(k, value)


if __name__ == "__main__":
    # get_query_embed()
    # merge_query_embed()
    # test_embed_keys_and_nums()
    # check_duplicated_query()
    check_each_index_has_five_query()
