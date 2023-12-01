import os
import openai
import pinecone


class TryEmptyInput(Exception):
    """
    入力が空の場合に発生する例外。
    """
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
    
    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.message}"



class Hippocampus:
    """
    自然言語によって記憶・回顧する。
    Pineconeと連携し、AIの海馬として振る舞う。
    """

    def __init__(self) -> None:
        # Pineconeの設定を.envから読み込む。
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set.")
        if not PINECONE_ENVIRONMENT:
            raise ValueError("PINECONE_ENVIRONMENT is not set.")

        pinecone.init(api_key=PINECONE_API_KEY,
                      environment=PINECONE_ENVIRONMENT)

        TABLE_NAME = os.getenv("TABLE_NAME", "play_with_gpt")
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        if TABLE_NAME not in pinecone.list_indexes():
            pinecone.create_index(
                TABLE_NAME, dimension=dimension, metric=metric, pod_type=pod_type
            )
        self.index = pinecone.Index(TABLE_NAME)

    def text_to_vector(self, text: str) -> list[float]:
        """
        自然言語テキストをadaによってベクトル化する。

        Args:
            text (str): 自然言語

        Returns:
            list[float]: ベクトル化された自然言語
        """
        
        if not text:
            raise TryEmptyInput("Input is empty.")

        # text = text.replace("\n", " ")
        response = openai.Embedding.create(  # type: ignore[no-untyped-call]
            input=[text],
            model="text-embedding-ada-002"
        )
        result: list[float] = response["data"][0]["embedding"]
        return result

    def input_memory(self, id: str, input: str, namespace: str = "") -> None:
        """
        自然言語で記憶する。

        Args:
            id (str): 記憶のid
            input (str): 記憶する自然言語
            namespace (str): 記憶のnamespace
        """
        try:
            vector = self.text_to_vector(input)
        except TryEmptyInput as e:
            raise e
        vectors = [
            {
                "id": id,
                "values": vector
            }
        ]

        self.index.upsert(vectors, namespace)

    def delete_memory(self, id: str, namespace: str ="") -> None:
        """
        記憶を削除する。

        Args:
            id (str): 削除する記憶のid
            namespace (str): 削除する記憶のnamespace
        """
        ids = [id]
        self.index.delete(ids=ids, namespace=namespace)

    def delete_all_memory(self) -> None:
        """
        全ての記憶を削除する。
        """
        self.index.delete(delete_all = True)

    def query_memory(self, query: str, top_k: int = 1, namespace: str ="") -> list[str]:
        """
        自然言語で記憶を検索し、候補数分のIDを返す。

        Args:
            query (str): 検索クエリ
            top_k (int): 検索結果の上位何件を返すか
            namespace (str): 検索対象のnamespace

        Returns:
            QueryResponse: 検索結果
        """

        vector = self.text_to_vector(query)

        memory = self.index.query(vector=vector, top_k=top_k, namespace=namespace)
        # memoryから、idだけのリストを抽出する。
        memory_ids: list[str] = [item.id for item in memory.matches]
        return memory_ids
