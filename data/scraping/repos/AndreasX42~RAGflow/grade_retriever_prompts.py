from langchain.prompts import PromptTemplate

template = """
    Given the question:
    {query}
    
    Here are some documents retrieved in response to the question:
    {result}
    
    And here is the answer to the question:
    {answer}
    
    GRADING CRITERIA: We want to know if the question can be directly answered with the provided documents and without providing any additional outside sources. Does the retrieved documents make it possible to answer the question?

    Your response should be as follows without providing any additional information:

    GRADE: (0 to 1) - grade '0' means it is impossible to answer the questions with the documents in any way, grade '1' means the question can be fully answered with the provided documents. The more aspects of the questions can be answered the higher the grade should be, with a maximum grade of 1.
    """

GRADE_RETRIEVER_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)
