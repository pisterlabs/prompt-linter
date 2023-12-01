from flask_restx import Namespace, Resource
from flask import request
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, LLMChain, SimpleSequentialChain
from langchain.chains.conversation.memory import ConversationBufferMemory

api = Namespace("generate", description="Test Endpoint")

OPENAI_API_KEY = 'API-KEY'

llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-4')
conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)


def generate_domain_specific_question(conversation_buf, subject):
    KnowledgeTemplate = "Can you generate a knowledge testing question on the topic {topic} ?. Only provide the question"
    KPrompt = PromptTemplate(
        input_variables=['topic'], template=KnowledgeTemplate)
    return conversation_buf(KPrompt.format(
        topic=subject))['response']


def eval_domain_specific_answer(conversation_buf, answer, question):
    check_answer_template = f"The candidate provided this answer {answer} for the question {question}. Can you rate his answer in 1-5 where 1 stands for wrong answer and 5 stands for correct answer ?. Only give the number"
    score = conversation_buf(check_answer_template)['response']
    return {"score": score}


@api.route("/domainspecificquestion")
class Question(Resource):
    def get(self):
        question = generate_domain_specific_question(conversation_buf,
                                                     ["Recursion"])
        return {"type": "SUBJECT_QUESTION", "question": question}

    def post(self):
        request_data = request.get_json()
        answer = request_data['answer']
        question = request_data['question']
        return eval_domain_specific_answer(conversation_buf, answer, question)
