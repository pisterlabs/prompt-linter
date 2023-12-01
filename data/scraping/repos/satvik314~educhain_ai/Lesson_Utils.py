from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain import LLMChain


# def generate_response(template,LLM_Model,mode,std,lesson,board,subtopic):
def generate_response(template, mode, std, lesson, board, subtopic):
    llm_model = ChatOpenAI(temperature=0.5)
    prompt = PromptTemplate(template=template, input_variables=['mode', 'std', 'lesson', 'board', 'subtopic'])
    chain = LLMChain(prompt=prompt, llm=llm_model)
    return chain.run(mode=mode, std=std, lesson=lesson, board=board, subtopic=subtopic)


template = """You are teacher for {mode} for {std} student and you have to teach him lesson {lesson} of the
    {board} science book for subtopic{subtopic}, give me motivation to learn this lesson then give introduction then explain core 
    concepts with examples"""
