import os

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

"""
임베딩을 사용하는 기능이 있습니다. 임베딩을 바로 사용하기기 보다는 Vector index를 만들기 위해 전달하는 인자로 많이 사용되고 
주로 OpenAI의 임베딩을 많이 사용하지만, 비용적 절감을 위해 허깅페이스에서 제공하는 임베딩을 사용해서 절약하는 방식도 추천됩니다.
"""

# 임베딩 모델
def openai_embedding():
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)

    text = "안녕하세요! 해변에 갈 시간입니다"
    text_embedding = embeddings.embed_query(text)

    print (f"임베딩 길이 : {len(text_embedding)}")
    print (f"샘플은 다움과 같습니다 : {text_embedding[:5]}...")

# 다중 문서 임베딩
def batch_embedding():
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)
    text_embedding = embeddings.embed_documents(
        [
            "Hi there!",
            "Oh, hello!",
            "What's your name?",
            "My friends call me World",
            "Hello World!"
        ]
    )
    print(f"임베딩 길이 : {len(text_embedding)}, {len(text_embedding[0])}")
    print(f"샘플은 다움과 같습니다 : {text_embedding[0][:5]}...")


# 허깅페이스 모델 센텐스 트랜스포머 임베딩 # pip install sentence_transformers
def huggingface_embedding():
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda'}, # 모델의 전달할 키워드 인수
        # encode_kwargs={'normalize_embeddings': False},  # 모델의 `encode` 메서드를 호출할 때 전달할 키워드 인수
    )
    text = "안녕하세요! 해변에 갈 시간입니다"
    text_embedding = hf_embeddings.embed_query(text)
    print (f"임베딩 길이 : {len(text_embedding)}")
    print (f"샘플은 다움과 같습니다 : {text_embedding[:5]}...")

if __name__=="__main__":
    openai_embedding()
    batch_embedding()
    huggingface_embedding()



