import dotenv
import langchain

import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

dotenv.load_dotenv()

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate

response = rag_chain.invoke("What is Task Decomposition?")
print(response)

# RagFusion pipeline
original_query = "What is Task Decomposition?"

prompt = ChatPromptTemplate(input_variables=['original_query'],
                            messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
                                                                                        template='You are a helpful assistant that generates multiple search queries based on a single input query.')),
                                      HumanMessagePromptTemplate(
                                          prompt=PromptTemplate(input_variables=['original_query'],
                                                                template='Generate multiple search queries related to: {question} \n OUTPUT (4 queries):'))])

print("The prompt is: ", prompt)

generate_queries = (
        prompt | ChatOpenAI(temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


ragfusion_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

langchain.debug = True

ragfusion_chain.input_schema.schema()

ragfusion_chain.invoke({"question": original_query})

model = "gpt-3.5-turbo"

from langchain.schema.runnable import RunnablePassthrough

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

full_rag_fusion_chain = (
        {
            "context": ragfusion_chain,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
)

full_rag_fusion_chain.input_schema.schema()

full_rag_fusion_chain.invoke({"question": "Tell me about AutoGPT"})
