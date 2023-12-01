from langchain.embeddings import BedrockEmbeddings
from numpy import dot
from numpy.linalg import norm

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1"
)

query_vector = embeddings.embed_query("飛行車の最高速度は？")

print(f"ベクトル化された質問： {query_vector[:5]}")

document_1_vector = embeddings.embed_query("飛行車の最高速度は時速150キロメートルです。")
document_2_vector = embeddings.embed_query("鶏肉を適切に下味をつけた後、中火で焼きながらたまに裏返し、外側は香ばしく中は柔らかく仕上げる。")

cos_sim_1 = dot(query_vector, document_1_vector) / (norm(query_vector) * norm(document_1_vector))
print(f"ドキュメント1と質問の類似度： {cos_sim_1}")
cos_sim_2 = dot(query_vector, document_2_vector) / (norm(query_vector) * norm(document_2_vector))
print(f"ドキュメント2と質問の類似度： {cos_sim_2}")
