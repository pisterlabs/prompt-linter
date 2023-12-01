import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
import tiktoken
import os
from prompt import Prompt


class GPTAgent:

    def __init__(self):

        self.custom_prompt = Prompt()

    def chat(self, question, history, search_index, model_name: str = 'gpt-3.5-turbo', verbose: bool = False):

        # similarity search
        docs = search_index.similarity_search(question, k=4)
        references = ''.join(f'{doc.page_content}\n' for doc in docs)

        prompt_template = self.custom_prompt.get_prompt()
        prompt_string = self.custom_prompt.get_prompt_string(
            history, question, references)

        if verbose:
            print(prompt_string)

        # Display cost
        self.calculate_usage_price(model_name, prompt_string)

        # Prediction
        llm = ChatOpenAI(model=model_name)
        answer_chain = LLMChain(
            llm=llm, prompt=prompt_template, verbose=verbose)
        response = answer_chain.run(
            question=question, references=references, history=history)

        history += f'User:\n{question}\nSupport:\n{response}\n'

        return response, history

    def load_embeddings(self, texts: list, embedding_model: str = 'text-embedding-ada-002', embedding_cache_path: str =
    'data/faiss_index'):

        embeddings = OpenAIEmbeddings(model=embedding_model)

        if os.path.exists(embedding_cache_path):
            print(f'Loading the pkl file: {embedding_cache_path}')

            with open(embedding_cache_path, 'rb') as f:
                db = pickle.load(f)

        else:
            db = FAISS.from_texts(texts, embeddings)

            # Store the database to a pickle file
            with open(embedding_cache_path, 'wb') as f:
                pickle.dump(db, f)

        return db

    def get_num_tokens_from_string(self, text: str, model_name: str) -> int:
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(text))

        return num_tokens

    def calculate_usage_price(self, model_name: str, prompt_string: str) -> None:
        prompt_price = {'gpt-4': 0.03,
                        'gpt-4-0314': 0.03,
                        'gpt-4-32k': 0.06,
                        'gpt-3.5-turbo': 0.002}

        token_count = self.get_num_tokens_from_string(
            prompt_string, model_name)
        cost = prompt_price[model_name] * token_count / 1000
        print(
            f'Model name: {model_name} | Token count: {token_count} | Cost: {cost}')
