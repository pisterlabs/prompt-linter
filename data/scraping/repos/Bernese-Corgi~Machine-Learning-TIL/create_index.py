import os
import dotenv
import openai
import pinecone
import pandas as pd

from openai.embeddings_utils import get_embeddings


dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_ENVIRONMENT = os.environ['PINECONE_ENVIRONMENT']
DEFAULT_INDEX_NAME = "openai"
EMBEDDING_MODEL = "text-embedding-ada-002"

def init_openai():
    openai.api_key = OPENAI_API_KEY

def init_pinecone(*, dimension: int, index_name: str = DEFAULT_INDEX_NAME) -> pinecone.Index:
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=dimension
        )

    return pinecone.Index(index_name=index_name)

def get_pinecone(index_name: str = DEFAULT_INDEX_NAME) -> pinecone.Index:
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )
    if index_name not in pinecone.list_indexes():
        raise Exception('파인콘 인덱스가 존재하지 않습니다.')

    return pinecone.Index(index_name=index_name)
    
# def get_pinecone_stats(index_name: str = DEFAULT_INDEX_NAME):
#     index = pinecone.Index(index_name=index_name)
#     stats = index.describe_index_stats()
#     dims = stats['dimension']
#     count = stats['namespaces']['']['vector_count']
    
#     return dims, count

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.head()
    # print(list(df.content_text.values), type(list(df.content_text.values)))
    df['content_vector'] = get_embeddings(list_of_text=list(df.content_text.values), engine=EMBEDDING_MODEL)
    df['vector_id'] = df.vector_id.apply(str)

    df.info(show_counts=True)
    
    return df

def create_index():
    init_openai()
    
    post_df = load_data('data/blog_posts.csv')
    
    # embeds = get_embeddings(list_of_text=post_df, engine=EMBEDDING_MODEL)
    # print(len(embeds))
    index = init_pinecone(dimension=len(post_df.content_vector[0]))
    print(index)

    # df_batcher = BatchGenerator(300)
    
    # for batch_df in df_batcher(post_df):
    #     index.upsert(
    #         vectors=zip(batch_df.vector_id, batch_df.content_vector),
    #         namespace="content"
    #     )
    #     # TODO print로 감싸기??
    #     index.describe_index_stats()

if __name__ == "__main__":
    create_index()