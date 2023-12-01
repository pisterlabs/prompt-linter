from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.question_gen.prompts import DEFAULT_SUB_QUESTION_PROMPT_TMPL


def build_ho3_sub_query_engine(ho3_index):
    
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    embed_model = LangchainEmbedding(OpenAIEmbeddings())

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
        )

    ho3_query_engine = ho3_index.as_query_engine(
        service_context=service_context,
        similarity_top_k=10)

    question_gen = LLMQuestionGenerator.from_defaults(
        prompt_template_str="""
            Follow the example, but instead of giving a question, always prefix the question 
            with: 'By first identifying and quoting the most relevant sources, '. 
            """
        + DEFAULT_SUB_QUESTION_PROMPT_TMPL,
    )

    final_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[
            QueryEngineTool(
                query_engine=ho3_query_engine,
                metadata=ToolMetadata(
                    name="homeowners_policy_documents",
                    description="insurance policy contract detailing coverage provisions.",
                ),
            )
        ],
        question_gen=question_gen,
        service_context=service_context,
        use_async=False,
    )
    return final_engine