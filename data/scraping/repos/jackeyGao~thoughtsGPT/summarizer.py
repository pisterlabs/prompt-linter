"""
这个文件暂时废弃了 
"""
from typing import List
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel
# from summarizer import Summarizer
from thoughts_gpt.core.embedding import FolderIndex
from thoughts_gpt.core.const import SUGGESTED_QUESTION_PREFIX
from thoughts_gpt.core.qa import get_suggested_questions
from thoughts_gpt.core.prompts import get_summarization_prompt


class SummarizerWithSources(BaseModel):
    answer: str
    suggested_questions: List[str]


def summarizer_folder(
    folder_index: FolderIndex,
    query: str, 
    llm: BaseChatModel,
    suggested_questions_limit: int = 5,
) -> SummarizerWithSources:
    """Summarizer a folder index

    Args:
        query (str): The query to search for.
        folder_index (FolderIndex): The folder index to search.
        return_all (bool): Whether to return all the documents from the embedding or
        just the sources for the answer.
        model (str): The model to use for the answer generation.
        **model_kwargs (Any): Keyword arguments for the model.

    Returns:
        SummarizerWithSources: The answer and the source documents.
    """

    chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        verbose=True,
        # map_prompt=map_prompt,
        # combine_prompt=get_reduce_template(suggested_questions_limit),
        prompt=get_summarization_prompt(suggested_questions_limit),
    )

    relevant_docs = folder_index.index.similarity_search(query, 10)

    summaries = chain.run(relevant_docs)

    suggested_questions = get_suggested_questions(summaries)

    answer = summaries.split(f"{SUGGESTED_QUESTION_PREFIX}:")[0]

    return SummarizerWithSources(
        answer=answer,
        suggested_questions=suggested_questions
    )

# def summarizer_with_summarizer(
#     folder_index: FolderIndex,
#     llm: BaseChatModel,
#     suggested_questions_limit: int = 5,
# ) -> SummarizerWithSources:
    
#     model = Summarizer()
#     db_result = folder_index.index._collection.get()

#     full_text = "".join(doc for doc in db_result["documents"])

#     answer = model(full_text, num_sentences=3)

#     return SummarizerWithSources(
#         answer=answer,
#         suggested_questions=[]
#     )
