from langchain.prompts import PromptTemplate

template = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

Your response should be as follows:

CORRECTNESS: (1,2,3,4 or 5) - grade 1 means the answer was completly incorrect, a higher grade towards 5 means the answer is more correct, does clarify more parts of the question and is more readable. The best grade is 5.
"""

GRADE_ANSWER_PROMPT_FAST = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)

template = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.

  You'll be given a function grading_function which you'll call for each provided context, question and answer to submit your reasoning and score for the correctness, comprehensiveness and readability of the answer. 

  Below is your grading rubric: 

- Correctness: Does the answer correctly answer the question.

- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. 

- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer. Rate from 0 (completely unreadable) to 1 (highly readable)

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

Your response should be as follows, do not provide any additional information.

CORRECTNESS: (1,2,3,4 or 5) - scale from 1 worst to 5 best
COMPREHENSIVENESS: (1,2,3,4 or 5) - scale from 1 worst to 5 best
READABILITY: (1,2,3,4 or 5) - scale from 1 worst to 5 best
"""

GRADE_ANSWER_PROMPT_5_CATEGORIES_5_GRADES_ZERO_SHOT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)


template = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.

  You'll be given a function grading_function which you'll call for each provided context, question and answer to submit your reasoning and score for the correctness, comprehensiveness and readability of the answer. 

  Below is your grading rubric: 

- CORRECTNESS: If the answer correctly answer the question, below are the details for different scores:

  - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer.

      - For example, when asked “How to terminate a databricks cluster”, the answer is empty string, or content that’s completely irrelevant, or sorry I don’t know the answer.

  - Score 1: the answer provides some relevance to the question and answers one aspect of the question correctly.

      - Example:

          - Question: How to terminate a databricks cluster

          - Answer: Databricks cluster is a cloud-based computing environment that allows users to process big data and run distributed data processing tasks efficiently.

          - Or answer:  In the Databricks workspace, navigate to the "Clusters" tab. And then this is a hard question that I need to think more about it

  - Score 2: the answer mostly answer the question but is missing or hallucinating on one critical aspect.

      - Example:

          - Question: How to terminate a databricks cluster”

          - Answer: “In the Databricks workspace, navigate to the "Clusters" tab.

          Find the cluster you want to terminate from the list of active clusters.

          And then you’ll find a button to terminate all clusters at once”

  - Score 3: the answer correctly answer the question and not missing any major aspect

      - Example:

          - Question: How to terminate a databricks cluster

          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.

          Find the cluster you want to terminate from the list of active clusters.

          Click on the down-arrow next to the cluster name to open the cluster details.

          Click on the "Terminate" button. A confirmation dialog will appear. Click "Terminate" again to confirm the action.”

- COMPREHENSIVENESS: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. Below are the details for different scores:

  - Score 0: typically if the answer is completely incorrect, then the comprehensiveness is also zero score.

  - Score 1: if the answer is correct but too short to fully answer the question, then we can give score 1 for comprehensiveness.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: First, you will need a Databricks access token with the appropriate permissions. You can generate this token through the Databricks UI under the 'User Settings' option. And then (the rest is missing)

  - Score 2: the answer is correct and roughly answer the main aspects of the question, but it’s missing description about details. Or is completely missing details about one minor aspect.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.

  - Score 3: the answer is correct, and covers all the main aspects of the question

- READABILITY: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.

  - Score 0: the answer is completely unreadable, e.g. fully of symbols that’s hard to read; e.g. keeps repeating the words that it’s very hard to understand the meaning of the paragraph. No meaningful information can be extracted from the answer.

  - Score 1: the answer is slightly readable, there are irrelevant symbols or repeated words, but it can roughly form a meaningful sentence that cover some aspects of the answer.

      - Example:

          - Question: How to use databricks API to create a cluster?

          - Answer: You you  you  you  you  you  will need a Databricks access token with the appropriate permissions. And then then you’ll need to set up the request URL, then you can make the HTTP Request. Then Then Then Then Then Then Then Then Then

  - Score 2: the answer is correct and mostly readable, but there is one obvious piece that’s affecting the readability (mentioning of irrelevant pieces, repeated words)

      - Example:

          - Question: How to terminate a databricks cluster

          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.

          Find the cluster you want to terminate from the list of active clusters.

          Click on the down-arrow next to the cluster name to open the cluster details.

          Click on the "Terminate" button…………………………………..

          A confirmation dialog will appear. Click "Terminate" again to confirm the action.

  - Score 3: the answer is correct and reader friendly, no obvious piece that affect readability.

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

Your response should be as follows, do not provide any additional information.

CORRECTNESS: (0,1,2 or 3)
COMPREHENSIVENESS: (0,1,2 or 3)
READABILITY: (0,1,2 or 3)
"""

GRADE_ANSWER_PROMPT_3_CATEGORIES_4_GRADES_FEW_SHOT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)
