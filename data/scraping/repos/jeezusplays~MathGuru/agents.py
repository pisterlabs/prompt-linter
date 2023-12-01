from langchain.chains import LLMChain
from langchain.llms import OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain

from .settings import OPENAI_API_KEY

gpt35 = OpenAIChat(openai_api_key=OPENAI_API_KEY,temperature=0, model="gpt-3.5-turbo-0613")
gpt4 = OpenAIChat(openai_api_key=OPENAI_API_KEY,temperature=0, model="gpt-4-0613")


# Text to questions agent
template1 = """
Role:
You are a primary school math expert with advanced filtering capabilities.

OCR scanned text:
{ocr_text}

Task:
Process the following OCR scanned text. Your goal is to identify, segment, and list all the math-related questions that are complete and solvable with the given text alone. Exclude any questions that refer to figures, diagrams, or require additional information not included in the text. Organize the extracted questions into a structured format.

Constraints:
- Omit any questions that involve visual aids such as charts, tables, or pictures since these cannot be processed through text alone.
- Ensure each question is listed separately and includes all the information required to be answered.
- Exclude any text that does not constitute a math-related question suitable for primary school level.

Return strictly in this JSON format:
{{ "questions": list[list[string]] }}
"""

prompt1 = PromptTemplate(
    input_variables=["ocr_text"],
    template=template1                     
)

chain1 = LLMChain(
    llm=gpt35, # Using the stronger model for better extraction
    prompt=prompt1,
    output_key="question_list"
)

get_questions = SequentialChain(
    chains=[chain1],
    input_variables=["ocr_text"],
    output_variables=["question_list"],
    verbose=True
)


# Metadata Agent
template2 = """
Role:
You are an intelligent agent with specialization in primary school mathematics, tasked with evaluating math questions for their completeness and educational attributes.

Task:
Review each math question and determine its characteristics based on the provided primary school level. Your main goal is to generate a metadata report that states the level, the topics covered, the sufficiency of context for the question to be answerable, and the difficulty level.

Data Input:
{{
    "question": "{question_text}",
    "primary_level": {primary_level}
}}

Required Metadata Output:
Create a metadata summary using the structure shown below. The summary should accurately reflect the academic level, enumerate the covered math topics, assess the completeness of the question's context, and categorize the difficulty level.

Metadata Output Format:
{{
   "level": {primary_level},
   "topics": list[string],
   "enoughContext": bool,
   "difficulty": string ('Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard')
}}

Detailed Instructions for "enoughContext":
- A question has "enoughContext" (true) if it includes all necessary measurements, figures, diagrams, or data needed to reach an answer.
- If the question is missing any crucial element such as measurements for calculating volume, specific angles for geometry problems, or explicit diagrams where they are required for understanding, then "enoughContext" should be marked as false.
- Pay close attention to the language of the question. If it implies the need for additional information that is not provided in the text (like the need to reference a diagram that is not described or included), it does not have enough context.
- Questions that are open-ended without specific numerical or factual data required for a primary school math question should also be marked as lacking enough context.

Note:
- The determination of "enoughContext" is critical. It ensures the question is directly answerable from the information provided in the text. This should be a binary assessment—either all necessary information is present (true), or it is not (false).
- For any question marked with "enoughContext": false, include a brief explanation stating what crucial information is missing.
"""

prompt2 = PromptTemplate(
    input_variables=["question_text","primary_level"],
    template=template2                     
)

chain2 = LLMChain(
    llm=gpt35,
    prompt=prompt2,
    output_key="question_metadata"
)

get_metadata = SequentialChain(
    chains=[chain2],
    input_variables=["question_text","primary_level"],
    output_variables=["question_metadata"],
    verbose=True
)

# Clean and filter away questions that need figures, and stitch questions that are related together
template3 = """
Role: You are a primary school math expert tasked with refining math questions.
Data:
{question_text}
Task: From the provided OCR texts, perform the following actions:
1. Remove any questions that refer to figures, diagrams or does not have enough context.
2. Stitch together questions that are related or split apart.
3. Correct the sentence of all english errors.
4. Return the cleaned questions in a structured format.
{{ "questions": list[list[str]] }}
"""

prompt3 = PromptTemplate(
    input_variables=["question_text"],
    template=template3                     
)

chain3 = LLMChain(
    llm=gpt35,
    prompt=prompt3,
    output_key="refined_questions"
)
clean_questions = SequentialChain(
    chains=[chain3],
    input_variables=["question_text"],
    output_variables=["refined_questions"],
    verbose=True
)

# Regenerate question agent ===== TODO =====
template4 = """
Role:
1. You are a primary school math expert and a creative writer.
2. Your goal is to craft a new math question, ensuring that the question's core concept remains consistent with the provided metadata.
3. You should be innovative, incorporating different variables, numbers, or context to make the new question distinct from the original.
4. Ensure the generated question maintains the educational value and is mathematically sound.

Data:
{{
    "original_question": "{question_text}",
    "metadata": {question_metadata}
}}

Task:
1. Design a fresh math question based on the given metadata. 
2. Ensure the new question doesn't replicate the original one in terms of variables and context.
3. Validate the mathematical integrity of the question.
4. If the original question has multiple sentences, ensure the newly crafted question is coherent.

{{ "question": list[str] }}
"""

prompt4 = PromptTemplate(
    input_variables=["question_text","question_metadata"],
    template=template4                     
)

chain4 = LLMChain(
    llm=gpt35,
    prompt=prompt4,
    output_key="regenerated_question"
)

generate_question = SequentialChain(
    chains=[chain4],
    input_variables=["question_text","question_metadata"],
    output_variables=["regenerated_question"],
    verbose=True
)


