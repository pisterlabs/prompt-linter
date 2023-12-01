from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate


from langchain.chains.conversation.memory import ConversationBufferMemory

OPENAI_API_KEY = '#OPEN_API_KEY'
llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-4')
conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)


def generate_kg_question(conversation_buf):
    KnowledgeTemplate = "Can you generate a knowledge testing question on the topic {topic} ?. Only provide the question"
    KPrompt = PromptTemplate(
        input_variables=['topic'], template=KnowledgeTemplate)
    question = conversation_buf(KPrompt.format(
        topic="Knowledge Graphs"))['response']
    print(f"This is the question: {question}")
    return question


def get_kg_answer(conversation_buf, question):
    # TODO: Supply answer here
    answer = "Yo KGs stand for Know Gaming"
    return answer


def eval_kg_answer(conversation_buf, answer):
    check_answer_template = f"The candidate provided this answer {answer} for the above question. Can you rate his answer in 1-5 where 1 stands for wrong answer and 5 stands for correct answer ?. Only give the number"
    score = conversation_buf(check_answer_template)['response']
    print(score)


def run_kg_questions():
    question = generate_kg_question(conversation_buf)
    answer = get_kg_answer(conversation_buf, question)
    eval_kg_answer(conversation_buf, answer)


run_kg_questions()
