from langchain.llms import OpenAI
from langchain.chains import ConversationChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate


from langchain.chains.conversation.memory import ConversationBufferMemory

OPENAI_API_KEY = '#OPEN_API_KEY'
llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-4')
conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

def generate_behave_question(conversation_buf, topics, difficulty):
    topics = ", ".join(topics)
    CodingQuesTemplate = "Can you create an interview question on the value {topic} ?. Make the question {difficulty} and related to Software Developer. Only provide the question"
    # TestCaseTemplate = "Can you write the test cases for the above question ?"
    CodingPrompt = PromptTemplate(
        input_variables=['topic', 'difficulty'], template=CodingQuesTemplate)
    question = conversation_buf(CodingPrompt.format(
        topic=topics, difficulty=difficulty))['response']
    # test_cases = conversation_buf(TestCaseTemplate)['response']
    print(f"This is the question: {question}")
    # print(f"This is the test_case: {test_cases}")
    return question



def get_behave_answer(conversation_buf, question):
    # TODO: Supply answer here
    answer = """
    I believe in upholding strong ethical standards in the workplace. In a previous position, I encountered a situation where a colleague asked me to manipulate some data to present a more favorable outcome for a project. Recognizing the importance of integrity, I approached my colleague and explained that I couldn't compromise on ethical principles. I suggested alternative solutions that would maintain accuracy and transparency in our work. While this decision resulted in additional effort, it demonstrated my commitment to upholding integrity and fostering a culture of trust within the team.
    """
    return answer


def eval_behave_answer(conversation_buf, answer):
    check_answer_template = f"The candidate provided this answer {answer}. Can you provide an integer rating between 1 to 10 based on the question asked."
    get_answer_scores = f'Extract the scores from above into a csv string?'

    response = conversation_buf(check_answer_template)['response']
    print(response)
    scores = conversation_buf(get_answer_scores)['response']
    print(scores)
    summarize_review = f'Provide a summary of the {response}'
    summary = conversation_buf(summarize_review)['response']
    print(summary)


def run_behave_questions():
    question = generate_behave_question(
        conversation_buf, ['Integrity'], 'hard')
    # question, test_cases = generate_coding_question(
    #    conversation_buf, ['LinkedList', 'HashMap'], 'hard')
    answer = get_behave_answer(conversation_buf, question)
    eval_behave_answer(conversation_buf, answer)


run_behave_questions()
