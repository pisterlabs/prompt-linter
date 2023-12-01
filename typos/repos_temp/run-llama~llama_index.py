"""
The original question is given below.
This question has been translated into a SQL query. \
Both the SQL query and the response are given below.
Given the SQL response, the question has also been translated into a vector store query.
The vector store query and response is given below.
Given SQL query, SQL response, transformed vector store query, and vector store \
response, please synthesize a response to the original question.

Original question: {query_str}
SQL query: {sql_query_str}
SQL response: {sql_response_str}
Transformed vector store query: {query_engine_query_str}
Vector store response: {query_engine_response_str}
Response:
"""f"""def get_{function_field}_field(text: str):
    \"""
    Function to extract {field}.
    \"""
    {response!s}
""""""\
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:

{schema_str}

The query string should contain only text that is expected to match the contents of \
documents. Any conditions in the filter should not be mentioned in the query as well.

Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters take into account the descriptions of attributes.
Make sure that filters are only used as needed. If there are no filters that should be \
applied return [] for the filter value.\

If the user's query explicitly mentions number of documents to retrieve, set top_k to \
that number, otherwise do not set top_k.

""""""
<< Example 2. >>
Data Source:
```json
{info_str}
```

User Query:
{query_str}

Structured Request:
""""""\
You are a world class state of the art agent.

You have access to multiple tools, each representing a different data source or API.
Each of the tools has a name and a description, formatted as a JSON dictionary.
The keys of the dictionary are the names of the tools and the values are the \
descriptions.
Your purpose is to help answer a complex user question by generating a list of sub \
questions that can be answered by the tools.

These are the guidelines you consider when completing your task:
* Be as specific as possible
* The sub questions should be relevant to the user question
* The sub questions should be answerable by the tools provided
* You can generate multiple sub questions for each tool
* Tools must be specified by their name, not their description
* You don't need to use a tool if you don't think it's relevant

Output the list of sub questions by calling the SubQuestionList function.

## Tools
```json
{tools_str}
```

## User Question
{query_str}
""""""
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query,
- a reference answer, and
- a generated answer.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, \
you should give a score of 1.
- If the generated answer is relevant but contains mistakes, \
you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, \
you should give a score between 4 and 5.

Example Response:
4.0
The generated answer has the exact same metrics as the reference answer, \
    but it is not as concise.

""""""
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""