import io
import os

from langchain import LLMChain
from langchain.chains import RetrievalQA, StuffDocumentsChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               PromptTemplate)
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


class FilesService:
    @staticmethod
    def _openai_streamer(retr_qa: RetrievalQA, text: str):
        for resp in retr_qa.run(text):
            yield resp

    @staticmethod
    async def query(question, temperature, n_docs, vectorstore):
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
            streaming=True,
            temperature=temperature,
        )
        messages = [
            SystemMessage(
                content="You are a world class algorithm to answer questions."
            ),
            HumanMessage(
                content="Answer question using only information contained in the following context: "
            ),
            HumanMessagePromptTemplate.from_template("{context}"),
            HumanMessage(
                content="Tips: If you can't find a relevant answer in the context, then say you don't know. Be concise!"
            ),
            HumanMessagePromptTemplate.from_template("Question: {question}"),
        ]
        prompt = ChatPromptTemplate(messages=messages)

        qa_chain = LLMChain(llm=llm, prompt=prompt)
        doc_prompt = PromptTemplate(
            template="Content: {page_content}",
            input_variables=["page_content"],
        )
        final_qa_chain = StuffDocumentsChain(
            llm_chain=qa_chain,
            document_variable_name="context",
            document_prompt=doc_prompt,
        )
        retrieval_qa = RetrievalQA(
            retriever=vectorstore.as_retriever(search_kwargs={"k": n_docs}),
            combine_documents_chain=final_qa_chain,
        )
        return FilesService._openai_streamer(retrieval_qa, question)

    @staticmethod
    async def upload(file, chunk_size, vectorstore):
        data = await file.read()
        elements = partition_pdf(file=io.BytesIO(data))
        text = [ele.text for ele in elements]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=20,
            length_function=len,
            add_start_index=True,
        )

        docs = text_splitter.create_documents(["\n".join(text)])
        response = vectorstore.add_documents(docs)
        return response
