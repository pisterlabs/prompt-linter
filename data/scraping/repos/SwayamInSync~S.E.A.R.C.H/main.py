import io
import contextlib
from llama_index.llms import ChatMessage, MessageRole
from llama_index import GPTVectorStoreIndex
from llama_index.indices.postprocessor import KeywordNodePostprocessor
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext

from tools import search_the_web
from utils import convert_documents_into_nodes
from embedding import EmbedNodes
from keyword_extraction import extract_keywords_vanilla
from reranker import cohere_rerank
from llms.zephyr import llm
from config import config

import nest_asyncio
nest_asyncio.apply()

embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"device": "cuda", "batch_size": 100})
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model)


def sub_query(query, index, service_context):
    vector_tool = QueryEngineTool(
        index.as_query_engine(),
        metadata=ToolMetadata(
            name="vector_search",
            description="Useful for searching for specific and correct facts."
        )
    )
    query_engine = SubQuestionQueryEngine.from_defaults([vector_tool],
                                                        service_context=service_context,
                                                        verbose=True)
    res = query_engine.query(query)
    return res


def only_llm(query):
    message = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Always answer the question correctly"
            ),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                f"answer the question: {query}\n"
            ),
        )]

    return llm.chat(message).message.content


def gen_code(query):
    message = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are a Python programmar, you write functionally correct and good python codes"
            ),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                f"Write python code for: {query}\n"
            ),
        )]
    res = llm.chat(message).message.content

    output = io.StringIO()
    code_to_execute = res.split("```")[1][6:].strip()
    with contextlib.redirect_stdout(output):
        exec(code_to_execute)

    captured_output = output.getvalue()
    output.close()

    return captured_output, code_to_execute


def get_answer(query, extra_args):
    include_images, improve_response = extra_args

    loaded_docs = search_the_web(query, include_images)

    nodes = convert_documents_into_nodes(loaded_docs)
    embedded_nodes = EmbedNodes()(nodes)

    index = GPTVectorStoreIndex(
        nodes=embedded_nodes, service_context=service_context)

    req_keywords = extract_keywords_vanilla(query)
    keyword_processor = KeywordNodePostprocessor(
        required_keywords=req_keywords)

    if improve_response:
        res = sub_query(query, index, service_context)
    else:
        index_query_engine = index.as_query_engine(
            similarity_top_k=config.retriver_top_k, node_postprocessors=[cohere_rerank, keyword_processor])
        res = index_query_engine.query(query)

    refs = []
    images = set()

    for node in res.source_nodes:
        if len(node.metadata) > 0:
            refs.append(node.metadata.get('URL'))
            if node.metadata.get("images") != None:
                for img in node.metadata.get("images"):
                    images.add(img)

    images = list(images)

    output = {
        "content": res.response,
        "images": images,
        "references": refs
    }
    return output


def process(query, extra_args):
    is_web, is_code, is_img_query, is_sub_query = extra_args

    if is_code:
        ans = gen_code(query)
        op, code = ans
        content = f"Possible output: {op}<br>Code:<br><code>{code}</code>"
        output = {
            "content": content,
            "images": [],
            "references": []
        }
        return output

    if not is_web:
        ans = only_llm(query)
        output = {
            "content": ans,
            "images": [],
            "references": []
        }
        return output

    return get_answer(query, (is_img_query, is_sub_query))


if __name__ == "__main__":
    query = input("Enter the query please")
    res = process(query, (True, False, False, False))
    print(res)
