from typing import Optional

from langchain import LLMChain, PromptTemplate, FAISS

from langchain.schema.language_model import BaseLanguageModel


def fact_checking(
    inputs: str, retriver: FAISS, llm: Optional[BaseLanguageModel] = None
):
    """
    The fact_checking function takes in a user query and returns the answer to that query.
        The function first checks if the hypothesis is grounded and entailed to the evidence.
        If it is, then it returns the original hypothesis as an answer. Otherwise, it generates a new response using only
        information provided in evidence section.

    Args:
        inputs: str: Pass the user's input to the fact_checking function
        retriver: FAISS: Retrieve the evidence from the database
        llm: Optional[BaseLanguageModel]: Pass the language model to the function

    Returns:
        A dictionary with two keys:
    """
    fact = retriver.similarity_search(inputs)
    evidence = ""
    for i in range(2 if len(fact)>2 else len(fact)):
        evidence = (
            evidence + "Evidence " + str(i) + ": " + fact[i].page_content + ". \n"
        )

    template = PromptTemplate(
        template="""
    You are given a task to identify if the hypothesis is grounded and entailed to the evidence.
    You will only use the contents of the evidence and not rely on external knowledge.
    You dont need to provide additional information in answer.
    Directly answer with a yes or no.
    \n
    "evidence":
    {evidence}
    \n\n
    "hypothesis":
    {hypothesis}
    """,
        input_variables=["evidence", "hypothesis"],
    )
    fact_check_chain = LLMChain(prompt=template, llm=llm)
    check = fact_check_chain.predict(hypothesis=inputs, evidence=evidence)
    check = check.lower().strip()
    print(check)
    if "no" in check and "yes" not in check:
        template = PromptTemplate(
            template="""
Generate a complete and accurate response to a user's query using only the information provided in the
 evidence section. Do not rely on external knowledge or information.
                "user query":
                {user_question}
                "evidence":
                {evidence}
                \n
                    """,
            input_variables=["evidence", "user_question"],
        )
        responsechain = LLMChain(prompt=template, llm=llm)
        answer = responsechain.predict(user_question=inputs, evidence=evidence)

        return {"check": "pass with new generated", "content": answer}

    else:
        return {"check": "pass", "content": inputs}
