from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from app.services.content_matching.service import ContentMatchingService
import json
import os
from typing import List, TypedDict


anthropic = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

def generate_question_extraction_prompt(base_questions_as_text: str) -> str:
 return f"{HUMAN_PROMPT}Extract questions as JSON array of strings from text: {base_questions_as_text}{AI_PROMPT}["

def generate_new_question_based_on_content_prompt(content: str, question: str) -> str:
  return f"""{HUMAN_PROMPT}
    <document>{content}</document>
    <question>{question}</question>
    Above is a document and a question related to that document. 
    Please write a new question that is related to the document in the same style.
    do not include the document or the original question in your answer.
    {AI_PROMPT}
    Here are the questions:
"""


def generate_new_question_with_answer_prompt(content: str, question: str) -> str:
  return f"""{HUMAN_PROMPT}
    <document>{content}</document>
    <question>{question}</question>
    Above is a document and a question related to that document. 
    You are to generate new questions and answers related to the document in the same style.
    return the result as a JSON object with the following structure:
    <json>
    {{"original_question": "What is the meaning of life?", "new_question": "What is the meaning of life?", "answer": "42"}}
    </json>
    {AI_PROMPT}{{
"""

class NewQuestion(TypedDict):
    original_question: str
    new_question: str
    answer: str

class QuestionGeneratorService:
  content_matching_service: ContentMatchingService

  def __init__(self):
    self.content_matching_service = ContentMatchingService()
  
  def generate_new_questions_with_answers(self, base_questions_as_text: str) -> List[NewQuestion]:
    questions = self.__parse_questions_from_text(base_questions_as_text)
    
    question_content_pairs = []
    for question in questions:
      print("Asking for question", question)
      content = self.content_matching_service.get_content_related_to_question(question)
      question_content_pairs.append({"question": question, "content": content})
    
    new_questions = []
    for pair in question_content_pairs:
      new_question = self.__generate_new_question_with_answer(pair["content"], pair["question"])
      new_questions.append(new_question)

    return new_questions


  def generate_new_questions(self, base_questions_as_text: str) -> List[str]:
    questions = self.__parse_questions_from_text(base_questions_as_text)
    
    question_content_pairs = []
    for question in questions:
      print("Asking for question", question)
      content = self.content_matching_service.get_content_related_to_question(question)
      print(content)
      question_content_pairs.append({"question": question, "content": content})
    
    new_questions = []
    for pair in question_content_pairs:
      new_question = self.__generate_new_question(pair["content"], pair["question"])
      new_questions.append(new_question)
    return "\n".join(new_questions)
  
  def __parse_questions_from_text(self, text: str) -> List[str]:
    completion = anthropic.completions.create(
        model="claude-2",
        temperature=0,
        max_tokens_to_sample=5000,
        prompt=generate_question_extraction_prompt(base_questions_as_text=text)
    )
    print("GOT COMPLETION", completion.completion)
    return json.loads("[" + completion.completion)
  

  def __generate_new_question(self, content: str, question: str) -> str:
    completion = anthropic.completions.create(
        model="claude-2",
        temperature=0,
        max_tokens_to_sample=5000,
        prompt=generate_new_question_based_on_content_prompt(content=content, question=question)
    )
    print(completion)
    return completion.completion
  
  def __generate_new_question_with_answer(self, content: str, question: str) -> NewQuestion:
    completion = anthropic.completions.create(
        model="claude-2",
        temperature=0,
        max_tokens_to_sample=5000,
        prompt=generate_new_question_with_answer_prompt(content=content, question=question)
    )
    print(completion)
    parsed = json.loads("{" + completion.completion)
    return NewQuestion(
      original_question=parsed["original_question"],
      new_question=parsed["new_question"],
      answer=parsed["answer"]
    )
  
