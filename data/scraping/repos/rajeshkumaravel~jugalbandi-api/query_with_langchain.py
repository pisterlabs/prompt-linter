import openai
from gpt_index import SimpleDirectoryReader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, OpenAI, LLMChain
from cloud_storage import *
import shutil
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('jugalbandi_api')

def langchain_indexing(uuid_number):
    sources = SimpleDirectoryReader(uuid_number).load_data()
    source_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=4 * 1024, chunk_overlap=200)
    counter = 0
    for source in sources:
        for chunk in splitter.split_text(source.text):
            new_metadata = {"source": str(counter)}
            source_chunks.append(Document(page_content=chunk, metadata=new_metadata))
            counter += 1
    try:
        search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())
        search_index.save_local("")
        error_message = None
        status_code = 200
    except openai.error.RateLimitError as e:
        error_message = f"OpenAI API request exceeded rate limit: {e}"
        status_code = 500
    except (openai.error.APIError, openai.error.ServiceUnavailableError):
        error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
        status_code = 503
    except Exception as e:
        error_message = str(e.__context__) + " and " + e.__str__()
        status_code = 500
    return error_message, status_code


def rephrased_question(user_query):
    template = """
    Write the same question as user input and make it more descriptive without adding new information and without making the facts incorrect.

    User: {question}
    Rephrased User input:"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=False)
    response = llm_chain.predict(question=user_query)
    return response.strip()


def querying_with_langchain(uuid_number, query):
    files_count = read_langchain_index_files(uuid_number)
    if files_count == 2:
        try:
            search_index = FAISS.load_local(uuid_number, OpenAIEmbeddings())
            chain = load_qa_with_sources_chain(
                OpenAI(temperature=0), chain_type="map_reduce"
            )
            paraphrased_query = rephrased_question(query)
            documents = search_index.similarity_search(paraphrased_query, k=5)
            answer = chain(
                {"input_documents": documents, "question": paraphrased_query}
            )
            answer_list = answer["output_text"].split("\nSOURCES:")
            final_answer = answer_list[0].strip()
            source_ids = answer_list[1]
            source_ids = source_ids.replace(" ", "")
            source_ids = source_ids.replace(".", "")
            source_ids = source_ids.split(",")
            final_source_text = ""
            for document in documents:
                if document.metadata["source"] in source_ids:
                    final_source_text += document.page_content + "\n\n"
            shutil.rmtree(uuid_number)
            return final_answer, final_source_text, paraphrased_query, None, 200
        except openai.error.RateLimitError as e:
            error_message = f"OpenAI API request exceeded rate limit: {e}"
            status_code = 500
        except (openai.error.APIError, openai.error.ServiceUnavailableError):
            error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
            status_code = 503
        except Exception as e:
            error_message = str(e.__context__) + " and " + e.__str__()
            status_code = 500
    else:
        error_message = "The UUID number is incorrect"
        status_code = 422
    return None, None, None, error_message, status_code


def querying_with_langchain_gpt4(uuid_number, query):
    if uuid_number.lower() == "storybot":
        try:
            system_rules = "I want you to act as an Indian story teller. You will come up with entertaining stories that are engaging, imaginative and captivating for children in India. It can be fairy tales, educational stories or any other type of stories which has the potential to capture children’s attention and imagination. A story should not be more than 200 words. The audience for the stories do not speak English natively. So use very simple English with short and simple sentences, no complex or compound sentences. Extra points if the story ends with an unexpected twist."
            res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_rules},
                    {"role": "user", "content": query},
                ],
            )
            return res["choices"][0]["message"]["content"], "", "", None, 200
        except openai.error.RateLimitError as e:
            error_message = f"OpenAI API request exceeded rate limit: {e}"
            status_code = 500
        except (openai.error.APIError, openai.error.ServiceUnavailableError):
            error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
            status_code = 503
        except Exception as e:
            error_message = str(e.__context__) + " and " + e.__str__()
            status_code = 500
        return None, None, None, error_message, status_code
    else:
        files_count = read_langchain_index_files(uuid_number)
        if files_count == 2:
            try:
                search_index = FAISS.load_local(uuid_number, OpenAIEmbeddings())
                documents = search_index.similarity_search(query, k=5)
                contexts = [document.page_content for document in documents]
                augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query
                system_rules = "You are a helpful assistant who helps with answering questions based on the provided information. If the information cannot be found in the text provided, you admit that I don't know"

                res = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_rules},
                        {"role": "user", "content": augmented_query},
                    ],
                )
                return res["choices"][0]["message"]["content"], "", "", None, 200

            except openai.error.RateLimitError as e:
                error_message = f"OpenAI API request exceeded rate limit: {e}"
                status_code = 500
            except (openai.error.APIError, openai.error.ServiceUnavailableError):
                error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
                status_code = 503
            except Exception as e:
                error_message = str(e.__context__) + " and " + e.__str__()
                status_code = 500
        else:
            error_message = "The UUID number is incorrect"
            status_code = 422
        return None, None, None, error_message, status_code

def querying_with_langchain_gpt4_streaming(uuid_number, query):
    files_count = read_langchain_index_files(uuid_number)
    if files_count == 2:
        try:
            search_index = FAISS.load_local(uuid_number, OpenAIEmbeddings())
            documents = search_index.similarity_search(query, k=5)
            contexts = [document.page_content for document in documents]
            augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

            system_rules = "You are a helpful assistant who helps with answering questions based on the provided information. If the information cannot be found in the text provided, you admit that I don't know"

            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[
                    {"role": "system", "content": system_rules},
                    {"role": "user", "content": augmented_query}
                ],
                stream=True
            )

            # Define a generator function to yield each chunk of the response
            async def generate_messages():
                for chunk in response:
                    print(chunk)
                    # chunk_message = chunk['choices'][0]['delta']['content']
                    # chunk_message = chunk["choices"][0].get("delta", {}).get("content", '')
                    chunk_message = chunk["choices"][0].get("delta", {}).get("content", '')
                    yield chunk_message

            # Return a StreamingResponse with the generated messages
            return EventSourceResponse(generate_messages(), headers={"Content-Type":"text/plain"})
            # application/json

        except openai.error.RateLimitError as e:
            error_message = f"OpenAI API request exceeded rate limit: {e}"
            status_code = 500
            logger.exception("RateLimitError occurred: %s", e)
        except (openai.error.APIError, openai.error.ServiceUnavailableError):
            error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
            status_code = 503
            logger.exception("APIError or ServiceUnavailableError occurred")
        except Exception as e:
            error_message = str(e.__context__) + " and " + e.__str__()
            status_code = 500
            logger.exception("An exception occurred: %s", e)
    else:
        error_message = "The UUID number is incorrect"
        status_code = 422

    # return None, None, None, error_message, status_codewss
    # If there's an error, return a plain text response with the error message
    return Response(content=error_message, media_type="text/plain", status_code=status_code)
