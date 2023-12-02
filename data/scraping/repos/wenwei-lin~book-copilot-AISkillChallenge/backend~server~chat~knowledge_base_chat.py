from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs.model_config import (llm_model_dict, LLM_MODEL, PROMPT_TEMPLATE,
                                  VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, summarization_config, SUMMARIZATION_MODEL,
                                  SUMMARIZATION_TYPE)
from server.chat.utils import wrap_done
from server.utils import BaseResponse
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
import json
import os
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import tiktoken


def calculate_token_usage(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def knowledge_base_chat(query: str = Body(..., description="user's input", examples=["hello"]),
                        knowledge_base_name: str = Body(..., description="knowledge_input", examples=["samples"]),
                        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="number of search"),
                        score_threshold: float = Body(SCORE_THRESHOLD,
                                                      description="Knowledge Base Matching Relevance Threshold, value range between 0-1, the smaller the SCORE, the higher the relevance, to 1 is equivalent to not filtering, it is recommended to set at about 0.5",
                                                      ge=0, le=1),
                        history: List[History] = Body([],
                                                      description="history",
                                                      examples=[]
                                                      ),
                        stream: bool = Body(False, description="stream"),
                        local_doc_url: bool = Body(False, description="file name"),
                        request: Request = None,
                        ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"could no find {knowledge_base_name}")

    history = [History(**h) if isinstance(h, dict) else h for h in history]

    async def knowledge_base_chat_iterator(query: str,
                                           kb: KBService,
                                           top_k: int,
                                           history: Optional[List[History]],
                                           ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
        if LLM_MODEL == "Azure-OpenAI":
            model = AzureChatOpenAI(
                streaming=True,
                verbose=True,
                callbacks=[callback],
                openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
                openai_api_version=llm_model_dict[LLM_MODEL]["api_version"],
                deployment_name=llm_model_dict[LLM_MODEL]["deployment_name"],
                openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
                openai_api_type="azure",
            )
        else:
            model = ChatOpenAI(
                streaming=True,
                verbose=True,
                callbacks=[callback],
                openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
                openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
                model_name=LLM_MODEL
            )

        if SUMMARIZATION_TYPE:
            if SUMMARIZATION_MODEL == "Azure-openai":
                model_summary = AzureChatOpenAI(
                    streaming=True,
                    verbose=True,
                    callbacks=[callback],
                    openai_api_base=llm_model_dict[SUMMARIZATION_MODEL]["api_base_url"],
                    openai_api_version=llm_model_dict[SUMMARIZATION_MODEL]["api_version"],
                    deployment_name=llm_model_dict[SUMMARIZATION_MODEL]["deployment_name"],
                    openai_api_key=llm_model_dict[SUMMARIZATION_MODEL]["api_key"],
                    openai_api_type="azure",
                )
            else:
                model_summary = ChatOpenAI(
                    streaming=True,
                    verbose=True,
                    temperature=0,
                    openai_api_key=llm_model_dict[SUMMARIZATION_MODEL]["api_key"],
                    openai_api_base=llm_model_dict[SUMMARIZATION_MODEL]["api_base_url"],
                    model_name=SUMMARIZATION_MODEL
                )

        if SUMMARIZATION_MODEL == "Stuff":
            stuff_chain = StuffDocumentsChain(llm=model_summary,
                                              document_variable_name=summarization_config[SUMMARIZATION_MODEL][
                                                  "map_document_variable_name"])
            context = stuff_chain.run(docs)
        elif SUMMARIZATION_MODEL == "Map_Reduce":
            # Map
            map_template = summarization_config[SUMMARIZATION_MODEL]["map_prompt_template"]
            map_prompt = PromptTemplate.from_template(map_template)
            map_chain = LLMChain(llm=model_summary, prompt=map_prompt)

            # Reduce
            reduce_template = summarization_config[SUMMARIZATION_MODEL]["reduce_prompt_template"]
            reduce_prompt = PromptTemplate.from_template(reduce_template)
            reduce_chain = LLMChain(llm=model_summary, prompt=reduce_prompt)

            # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
            combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name=
            summarization_config[SUMMARIZATION_MODEL]["reduce_document_variable_name"])

            reduce_documents_chain = ReduceDocumentsChain(
                combine_documents_chain=combine_documents_chain,
                collapse_documents_chain=combine_documents_chain,
                token_max=summarization_config[SUMMARIZATION_MODEL]["token_max"],
            )

            # Combining documents by mapping a chain over them, then combining results
            map_reduce_chain = MapReduceDocumentsChain(
                llm_chain=map_chain,
                reduce_documents_chain=reduce_documents_chain,
                document_variable_name=summarization_config[SUMMARIZATION_MODEL]["map_document_variable_name"],
                return_intermediate_steps=False)
            context = map_reduce_chain.run(docs)
        elif SUMMARIZATION_MODEL == "Refine":
            refine_chain = load_summarize_chain(model_summary, chain_type="refine")
            context = refine_chain.run(docs)
        else:
            context = "\n".join([doc.page_content for doc in docs])
        history_use = [i.to_msg_tuple() for i in history]
        chat_use = [("human", PROMPT_TEMPLATE)]
        chat_prompt = ChatPromptTemplate.from_messages(history_use + chat_use)
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.

        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            text = f"""source: [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if stream:

            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token,
                                  "docs": source_documents},
                                 ensure_ascii=False)
        else:

            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)

        await task

    return StreamingResponse(knowledge_base_chat_iterator(query, kb, top_k, history),
                             media_type="text/event-stream")
