from flask_restx import Namespace, Resource
from flask import request
import json
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, LLMChain, SimpleSequentialChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from .sqlalchemy_engine import cursor, cnxn

api = Namespace("generate", description="Test Endpoint")

OPENAI_API_KEY = 'API-KEY'

llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-4')
conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)


def generate_coding_question(topics, difficulty):
    topics = ", ".join(topics)
    CodingQuesTemplate = "Can you create a coding question on the topic {topic} ?. Make the question {difficulty}. Only provide the question and nothing else without any special characters in the question text."
    # TestCaseTemplate = "Can you write the test cases for the above question ?"
    CodingPrompt = PromptTemplate(
        input_variables=['topic', 'difficulty'], template=CodingQuesTemplate)
    question = conversation_buf(CodingPrompt.format(
        topic=topics, difficulty=difficulty))['response']
    # test_cases = conversation_buf(TestCaseTemplate)['response']
    print(f"This is the question: {question}")
    # print(f"This is the test_case: {test_cases}")
    return question, ""


def eval_coding_answer(conversation_buf, answer, question):
    check_answer_template = f"The candidate provided this answer {answer} for this question {question}. Can you rate the answer based on code quality, correctness and code formatting"
    get_answer_scores = "Extract the scores from above into a csv string?"
    response = conversation_buf(check_answer_template)['response']
    print(response)
    scores = conversation_buf(get_answer_scores)['response']
    print(scores)
    summarize_review = f'Provide a summary of the {response}'
    summary = conversation_buf(summarize_review)['response']
    print(summary)
    return {"response": response, "scores": scores, "summary": summary}


@api.route("/codingquestion")
class Question(Resource):
    def get(self):
        question, test_cases = generate_coding_question(["Palindrome"], "hard")
        return {"type": "CODING_QUESTION", "question": question, "test_cases": test_cases}

    def post(self):
        request_data = request.get_json()
        interview_id = request_data["interview_id"]
        answer = request_data['answer']
        question = request_data['question']
        score_response = eval_coding_answer(conversation_buf, answer, question)
        count = cursor.execute("""INSERT INTO coding_stats VALUES (?, ?, ?, ?, ?)""",
                               interview_id, question, json.dumps({}), json.dumps(score_response), "vikrame")
        cnxn.commit()
        return score_response
