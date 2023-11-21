from langchain.chains.base import Chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough

prompt_template = """Write a concise summary of the following text, based on the user input.
User input: {query}
Text:
```
{text}
```
CONCISE SUMMARY:"""

refine_template = (
    "You are iteratively crafting a summary of the text below based on the user input\n"
    "User input: {query}\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary.\n"
    "If the context isn't useful, return the original summary.\n"
    "If the context is useful, refine the summary to include the new context.\n"
    "Your contribution is helping to build a comprehensive summary of a large body of knowledge.\n"
    "You do not have the complete context, so do not discard pieces of the original summary."
)


def get_summarization_chain(
    llm: BaseLanguageModel,
    prompt: str,
) -> Chain:
    _prompt = PromptTemplate.from_template(
        prompt_template,
        partial_variables={"query": prompt},
    )
    refine_prompt = PromptTemplate.from_template(
        refine_template,
        partial_variables={"query": prompt},
    )
    return load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=False,
        input_key="input_documents",
        output_key="output_text",
    )


def get_rag_summarization_chain(
    prompt: str,
    retriever: BaseRetriever,
    llm: BaseLanguageModel,
    input_key: str = "prompt",
) -> RunnableSequence:
    return (
        {"input_documents": retriever, input_key: RunnablePassthrough()}
        | get_summarization_chain(llm, prompt)
        | (lambda output: output["output_text"])
    )
