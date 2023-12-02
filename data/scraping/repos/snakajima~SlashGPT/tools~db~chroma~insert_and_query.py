import os

import chromadb
import numpy as np

# https://docs.trychroma.com/api-reference
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY", None)
EMBEDDING_MODEL = "text-embedding-ada-002"

# db_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../chroma-db"))
db_path = os.path.normpath(os.path.expanduser("~/.slashgpt/chroma-db"))

client = chromadb.PersistentClient(path=db_path)

collection = client.get_or_create_collection("test-database")

sampleDataSet = [
    "間違いを犯したことのない人とは、何も新しいことをしていない人だ。",
    "悪に感化される人が居る事よりも、悪を看過する人が居る事の方が危ない。",
    "何かを学ぶのに、自分自身で経験する以上に良い方法はない。",
    "想像力は知識よりも重要である。知識に限界があるが為に、想像力が世界をとりまき、発展を刺激しつづけ、進歩に息を吹き込みつづけているのだから。",
    "いつだって、偉大な先人達は凡人達の熾烈な抵抗に遭ってきた。",
    "人間性について絶望してはならない。なぜなら我々は人間なのだから。",
    "物事は全て、出来る限り単純にすべきだ。",
    "調べられるものを、いちいち覚えておく必要などない。",
    "一見して人生には何の意味もない。 しかし一つの意味もないということはあり得ない。",
    "人生を楽しむ秘訣は普通にこだわらないこと。 普通と言われる人生を送る人間なんて、一人としていやしない。 いたらお目にかかりたいものだ。",
    "学校で学んだことを一切忘れてしまった時に、なお残っているもの、それこそ教育だ。",
    "常識とは、18歳までに身に付けた偏見のコレクションである。",
    "私は将来について悩まない。すぐにやって来るから。",
]

i = 0
for query in sampleDataSet:
    query_embedding_response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    collection.upsert(
        ids=[str(i)],
        embeddings=[np.array(query_embedding_response.data[0].embedding).tolist()],
        metadatas=[
            {"id": i},
        ],
        documents=[query],
    )
    i = i + 1

# response = collection.get()
# print(response)

# q = "いつだって、偉大な先人達は凡人達の熾烈な抵抗に遭ってきた。",
q = "人間性について絶望してはならない。なぜなら我々は人間なのだから。"

query_embedding_response = openai.embeddings.create(
    model=EMBEDDING_MODEL,
    input=q,
)

res = collection.query(
    query_embeddings=[np.array(query_embedding_response.data[0].embedding).tolist()],
    n_results=1,
)

print(list(map(lambda x: "".join(x), list(*res["documents"]))))
