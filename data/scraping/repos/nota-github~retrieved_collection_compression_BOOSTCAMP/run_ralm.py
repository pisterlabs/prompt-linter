"""Ask a question to the notion database."""

import sys
import argparse
from typing import List

from langchain.chat_models import ChatOpenAI  # for `gpt-3.5-turbo` & `gpt-4`
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseRetriever, Document
import gradio as gr

from retrieve import Retriever

DEFAULT_QUESTION = "Wikipedia 2018 english dump에서 궁금한 점을 질문해주세요.\n예를들어 \n\n- Where are mucosal associated lymphoid tissues present in the human body and why?\n- When did korean drama started in the philippines?\n- When did the financial crisis in greece start?"
TEMPERATURE = 0


class LangChainCustomRetrieverWrapper(BaseRetriever):
    def __init__(self, args):
        self.args = args

        self.retriever = Retriever(args)  # DensePhrase

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get texts relevant for a query.

        Args:
            query: string to find relevant texts for

        Returns:
            List of relevant documents
        """

        print(f"query = {query}")

        # retrieve
        results = self.retriever.retrieve(single_query_or_queries_dict=query)

        # make result list of Document object
        return [
            Document(page_content=result, metadata={"source": f"source_{idx}"})
            for idx, result in enumerate(results)
        ]

    async def aget_relevant_documents(
        self, query: str
    ) -> List[Document]:  # abstractmethod
        raise NotImplementedError


class RaLM:
    def __init__(self, args):
        self.args = args
        self.initialize_ralm()

    def initialize_ralm(self):
        # initialize custom retriever
        self.retriever = LangChainCustomRetrieverWrapper(self.args)

        # prompt for RaLM
        system_template = """Use the following pieces of context to answer the users question.
        Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
        Always try to generate answer from source.
        ----------------
        {summaries}"""
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain_type_kwargs = {"prompt": prompt}
        llm = ChatOpenAI(model_name=self.args.model_name, temperature=TEMPERATURE)
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            reduce_k_below_max_tokens=True,
            chain_type_kwargs=chain_type_kwargs,
        )

    def run_chain(self, question, force_korean=False):
        if force_korean:
            question = f"{question} 본문을 참고해서 한글로 대답해줘"
        result = self.chain({"question": question})

        # postprocess
        result["answer"] = self.postprocess(result["answer"])
        if isinstance(result["sources"], str):
            result["sources"] = self.postprocess(result["sources"])
            result["sources"] = result["sources"].split(", ")
            result["sources"] = [src.strip() for src in result["sources"]]

        # print result
        self.print_result(result)

        return result

    def print_result(self, result):
        print(f"Answer: {result['answer']}")

        print(f"Sources: ")
        print(result["sources"])
        assert isinstance(result["sources"], list)
        nSource = len(result["sources"])

        for i in range(nSource):
            source_title = result["sources"][i]
            print(f"{source_title}: ")
            if "source_documents" in result:
                for j in range(len(result["source_documents"])):
                    if result["source_documents"][j].metadata["source"] == source_title:
                        print(result["source_documents"][j].page_content)
                        break

    def postprocess(self, text):
        # remove final parenthesis (bug with unknown cause)
        if (
            text.endswith(")")
            or text.endswith("(")
            or text.endswith("[")
            or text.endswith("]")
        ):
            text = text[:-1]

        return text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question to the notion DB.")

    # General
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo-16k-0613",
        help="model name for openai api",
    )

    # Retriever: Densephrase
    parser.add_argument(
        "--query_encoder_name_or_dir",
        type=str,
        default="princeton-nlp/densephrases-multi-query-multi",
        help="query encoder name registered in huggingface model hub OR custom query encoder checkpoint directory",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="start/1048576_flat_OPQ96_small",
        help="index name appended to index directory prefix",
    )

    args = parser.parse_args()

    # to prevent collision with DensePhrase native argparser
    sys.argv = [sys.argv[0]]

    # initialize class
    app = RaLM(args)

    def question_answer(question):
        result = app.run_chain(question=question, force_korean=False)

        return result[
            "answer"
        ], "\n######################################################\n\n".join(
            [
                f"Source {idx}\n{doc.page_content}"
                for idx, doc in enumerate(result["source_documents"])
            ]
        )

    # launch gradio
    gr.Interface(
        fn=question_answer,
        inputs=gr.inputs.Textbox(default=DEFAULT_QUESTION, label="질문"),
        outputs=[
            gr.inputs.Textbox(default="챗봇의 답변을 표시합니다.", label="생성된 답변"),
            gr.inputs.Textbox(
                default="prompt에 사용된 검색 결과들을 표시합니다.", label="prompt에 첨부된 검색 결과들"
            ),
        ],
        title="지식기반 챗봇",
        theme="dark-grass",
        description="사용자의 지식베이스에 기반해서 대화하는 챗봇입니다.\n본 예시에서는 wikipedia dump에서 검색한 후 이를 바탕으로 답변을 생성합니다.\n\n retriever: densePhrase, generator: gpt-3.5-turbo-16k-0613 (API)",
    ).launch(share=True)
