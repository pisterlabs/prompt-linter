import ctypes
from typing import Union, List
import numpy as np
from typing import Any, List, Optional
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import Field

N_THREADS = 6


class BertCppEmbeddings(Embeddings):
    client: Any  #: :meta private:
    model_path: str

    n_ctx: int = Field(512, alias="n_ctx")
    """Token context window."""

    n_threads: Optional[int] = Field(N_THREADS, alias="n_threads")
    """Number of threads to use. If None, the number 
    of threads is automatically determined."""

    n_batch: Optional[int] = Field(16, alias="n_batch")
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    def __init__(self, fname):
        self.lib = ctypes.cdll.LoadLibrary("libbert.so")

        self.lib.bert_load_from_file.restype = ctypes.c_void_p
        self.lib.bert_load_from_file.argtypes = [ctypes.c_char_p]

        self.lib.bert_n_embd.restype = ctypes.c_int32
        self.lib.bert_n_embd.argtypes = [ctypes.c_void_p]

        self.lib.bert_free.argtypes = [ctypes.c_void_p]

        self.lib.bert_encode_batch.argtypes = [
            ctypes.c_void_p,  # struct bert_ctx * ctx,
            ctypes.c_int32,  # int32_t n_threads,
            ctypes.c_int32,  # int32_t n_batch_size
            ctypes.c_int32,  # int32_t n_inputs
            ctypes.POINTER(ctypes.c_char_p),  # const char ** texts
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # float ** embeddings
        ]

        self.ctx = self.lib.bert_load_from_file(fname.encode("utf-8"))
        self.n_embd = self.lib.bert_n_embd(self.ctx)

    def __del__(self):
        self.lib.bert_free(self.ctx)

    def embed(self, sentences: Union[str, List[str]]) -> np.ndarray:
        input_is_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_is_string = True

        n = len(sentences)

        embeddings = np.zeros((n, self.n_embd), dtype=np.float32)
        embeddings_pointers = (ctypes.POINTER(ctypes.c_float) * len(embeddings))(
            *[e.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for e in embeddings]
        )

        texts = (ctypes.c_char_p * n)()
        for j, sentence in enumerate(sentences):
            texts[j] = sentence.encode("utf-8")

        self.lib.bert_encode_batch(
            self.ctx, 6, 16, len(sentences), texts, embeddings_pointers
        )
        if input_is_string:
            return embeddings[0]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [self.embed(text) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self.embed(text)
        return list(map(float, embedding))


# def main():
#     model_path = '../../models/ggml-model-q4_1.bin'
#     model = BertCppEmbeddings(model_path)

#     txt_file = "sample_client_texts.txt"
#     print(f"Loading texts from {txt_file}...")
#     with open(os.path.join(os.path.dirname(__file__), txt_file), 'r') as f:
#         texts = f.readlines()
#     print(datetime.datetime.now())
#     embedded_texts = model.embed_documents(texts)
#     print(datetime.datetime.now())

#     print(f"Loaded {len(texts)} lines.")

#     def print_results(res):
#         (closest_texts, closest_similarities) = res
#         # Print the closest texts and their similarity scores
#         print("Closest texts:")
#         for i, text in enumerate(closest_texts):
#             print(f"{i+1}. {text} (similarity score: {closest_similarities[i]:.4f})")

#     # Define the function to query the k closest texts
#     def query(text, k=3):
#         # Embed the input text
#         embedded_text = model.embed(text)
#         # Compute the cosine similarity between the input text and all the embedded texts
#         similarities = [np.dot(embedded_text, embedded_text_i) / (np.linalg.norm(embedded_text) * np.linalg.norm(embedded_text_i)) for embedded_text_i in embedded_texts]
#         # Sort the similarities in descending order
#         sorted_indices = np.argsort(similarities)[::-1]
#         # Return the k closest texts and their similarities
#         closest_texts = [texts[i] for i in sorted_indices[:k]]
#         closest_similarities = [similarities[i] for i in sorted_indices[:k]]
#         return closest_texts, closest_similarities

#     test_query = "Should I get health insurance?"
#     print(f'Starting with a test query "{test_query}"')
#     print_results(query(test_query))

#     while True:
#         # Prompt the user to enter a text
#         input_text = input("Enter a text to find similar texts (enter 'q' to quit): ")
#         # If the user enters 'q', exit the loop
#         if input_text == 'q':
#             break
#         # Call the query function to find the closest texts
#         print_results(query(input_text))


# if __name__ == '__main__':
#     main()
