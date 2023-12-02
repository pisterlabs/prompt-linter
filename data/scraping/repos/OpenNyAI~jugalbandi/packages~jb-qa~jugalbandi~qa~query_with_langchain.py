from typing import List
import openai
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, OpenAI, LLMChain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np
from jugalbandi.core.errors import (
    InternalServerException,
    ServiceUnavailableException
)
from jugalbandi.document_collection import DocumentCollection


async def rephrased_question(user_query: str):
    template = (
        """Write the same question as user input and """
        """make it more descriptive without adding """
        """new information and without making the facts incorrect.

    User: {question}
    Rephrased User input:"""
    )
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0),  # type: ignore
                         verbose=False)
    response = llm_chain.predict(question=user_query)
    return response.strip()


async def latent_semantic_analysis(response: str, documents: List):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    svd = TruncatedSVD(n_components=300)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    lsa_matrix = normalize(lsa_matrix)

    response_vector = vectorizer.transform([response])
    response_lsa = svd.transform(response_vector)
    response_lsa = normalize(response_lsa)

    scores = []
    for i, doc_lsa in enumerate(lsa_matrix):
        score = np.dot(response_lsa, doc_lsa.T)
        scores.append((i, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    similarity_scores = []
    for score in scores:
        scores_list = list(score[1])
        similarity_scores.append([score[0], scores_list[0]])

    return similarity_scores


async def querying_with_langchain(document_collection: DocumentCollection, query: str):
    await document_collection.download_index_files("langchain", "index.faiss",
                                                   "index.pkl")
    index_folder_path = document_collection.local_index_folder("langchain")
    try:
        search_index = FAISS.load_local(index_folder_path,
                                        OpenAIEmbeddings())  # type: ignore
        chain = load_qa_with_sources_chain(
            OpenAI(temperature=0), chain_type="map_reduce"  # type: ignore
        )
        paraphrased_query = await rephrased_question(query)
        documents = search_index.similarity_search(paraphrased_query, k=5)
        answer = chain({"input_documents": documents, "question": query})
        answer_list = answer["output_text"].split("\nSOURCES:")
        final_answer = answer_list[0].strip()
        source_ids = answer_list[1]
        source_ids = source_ids.replace(" ", "")
        source_ids = source_ids.replace(".", "")
        source_ids = source_ids.split(",")
        final_source_text = []
        for document in documents:
            if document.metadata["source"] in source_ids:
                final_source_text.append(document.page_content)
        return final_answer, final_source_text

    except openai.error.RateLimitError as e:
        raise ServiceUnavailableException(
            f"OpenAI API request exceeded rate limit: {e}"
        )
    except (openai.error.APIError, openai.error.ServiceUnavailableError):
        raise ServiceUnavailableException(
            "Server is overloaded or unable to answer your request at the moment."
            " Please try again later"
        )
    except Exception as e:
        raise InternalServerException(e.__str__())


async def querying_with_langchain_gpt4(document_collection: DocumentCollection,
                                       query: str,
                                       prompt: str):
    await document_collection.download_index_files("langchain", "index.faiss",
                                                   "index.pkl")
    index_folder_path = document_collection.local_index_folder("langchain")
    try:
        search_index = FAISS.load_local(index_folder_path,
                                        OpenAIEmbeddings())  # type: ignore
        documents = search_index.similarity_search(query, k=5)
        contexts = [document.page_content for document in documents]
        augmented_query = augmented_query = (
                "Information to search for answers:\n\n"
                "\n\n-----\n\n".join(contexts) +
                "\n\n-----\n\nQuery:" + query
            )

        if prompt != "":
            system_rules = prompt
        else:
            system_rules = (
                "You are a helpful assistant who helps with answering questions "
                "based on the provided information. If the information cannot be found "
                "in the text provided, you admit that I don't know"
            )
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": augmented_query},
            ],
        )
        return res["choices"][0]["message"]["content"], []

    except openai.error.RateLimitError as e:
        raise ServiceUnavailableException(
            f"OpenAI API request exceeded rate limit: {e}"
        )
    except (openai.error.APIError, openai.error.ServiceUnavailableError):
        raise ServiceUnavailableException(
            "Server is overloaded or unable to answer your request at the moment."
            " Please try again later"
        )
    except Exception as e:
        raise InternalServerException(e.__str__())


async def querying_with_langchain_gpt3_5(document_collection: DocumentCollection,
                                         query: str,
                                         prompt: str,
                                         source_text_filtering: bool,
                                         model_size: str):
    await document_collection.download_index_files("langchain", "index.faiss",
                                                   "index.pkl")
    index_folder_path = document_collection.local_index_folder("langchain")

    if model_size == "16k":
        model_name = "gpt-3.5-turbo-16k"
    else:
        model_name = "gpt-3.5-turbo"

    try:
        search_index = FAISS.load_local(index_folder_path,
                                        OpenAIEmbeddings())  # type: ignore
        documents = search_index.similarity_search(query, k=5)
        if prompt != "":
            system_rules = prompt
        else:
            system_rules = (
                "You are a helpful assistant who helps with answering questions "
                "based on the provided information. If the information cannot be found "
                "in the text provided, you admit that you don't know"
            )
        try:
            contexts = [document.page_content for document in documents]
            augmented_query = (
                "Information to search for answers:\n\n"
                "\n\n-----\n\n".join(contexts) +
                "\n\n-----\n\nQuery:" + query
            )
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_rules},
                    {"role": "user", "content": augmented_query},
                ],
            )
        except openai.error.InvalidRequestError:
            contexts = [documents[i].page_content for i in range(len(documents)-2)]
            augmented_query = (
                "Information to search for answers:\n\n"
                "\n\n-----\n\n".join(contexts) +
                "\n\n-----\n\nQuery:" + query
            )
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_rules},
                    {"role": "user", "content": augmented_query},
                ],
            )
        result = response["choices"][0]["message"]["content"]

        if source_text_filtering:
            files_dict = {}
            if len(documents) == 1:
                document = documents[0]
                if "txt_file_url" in document.metadata.keys():
                    source_text_link = document.metadata["txt_file_url"]
                    files_dict[source_text_link] = {
                        "source_text_link": source_text_link,
                        "source_text_name": document.metadata["document_name"],
                        "chunks": [document.page_content],
                    }
            else:
                similarity_scores = await latent_semantic_analysis(result, contexts)
                for score in similarity_scores:
                    if score[1] > 0.85:
                        document = documents[score[0]]
                        if "txt_file_url" in document.metadata.keys():
                            source_text_link = document.metadata["txt_file_url"]
                            if source_text_link not in files_dict:
                                files_dict[source_text_link] = {
                                    "source_text_link": source_text_link,
                                    "source_text_name": document.metadata[
                                        "document_name"
                                    ],
                                    "chunks": [],
                                }
                            content = document.page_content.replace("\\n", "\n")
                            files_dict[source_text_link]["chunks"].append(content)
            source_text_list = [files_dict[i] for i in files_dict]
        else:
            source_text_list = []
        return result, source_text_list

    except openai.error.RateLimitError as e:
        raise ServiceUnavailableException(
            f"OpenAI API request exceeded rate limit: {e}"
        )
    except (openai.error.APIError, openai.error.ServiceUnavailableError):
        raise ServiceUnavailableException(
            "Server is overloaded or unable to answer your request at the moment."
            " Please try again later"
        )
    except Exception as e:
        raise InternalServerException(e.__str__())
