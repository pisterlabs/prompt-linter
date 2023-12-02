from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.openai import embed_with_retry
from .Nl2Modelica import ModelObject
from typing import List, Optional
import numpy as np
class Nl2ModelEmbedding(OpenAIEmbeddings):

    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        try:
            import tiktoken

            tokens = []
            indices = []
            encoding = tiktoken.get_encoding(self.model)
            for i, text in enumerate(texts):
                if self.model.endswith("001"):
                    # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                    # replace newlines, which can negatively affect performance.
                    text = text.replace("\n", " ")
                token = encoding.encode(
                    text,
                    allowed_special=self.allowed_special,
                    disallowed_special=self.disallowed_special,
                )
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens += [token[j : j + self.embedding_ctx_length]]
                    indices += [i]

            batched_embeddings = []
            _chunk_size = chunk_size or self.chunk_size
            for i in range(0, len(tokens), _chunk_size):
                response = embed_with_retry(
                    self,
                    input=tokens[i : i + _chunk_size],
                    engine=self.deployment,
                    request_timeout=self.request_timeout,
                    headers=self.headers,
                )
                batched_embeddings += [r["embedding"] for r in response["data"]]

            results: List[List[List[float]]] = [[] for _ in range(len(texts))]
            num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
            for i in range(len(indices)):
                results[indices[i]].append(batched_embeddings[i])
                num_tokens_in_batch[indices[i]].append(len(tokens[i]))

            for i in range(len(texts)):
                _result = results[i]
                if len(_result) == 0:
                    average = embed_with_retry(
                        self,
                        input="",
                        engine=self.deployment,
                        request_timeout=self.request_timeout,
                        headers=self.headers,
                    )["data"][0]["embedding"]
                else:
                    average = np.average(
                        _result, axis=0, weights=num_tokens_in_batch[i]
                    )
                embeddings[i] = (average / np.linalg.norm(average)).tolist()

            return embeddings

        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )
