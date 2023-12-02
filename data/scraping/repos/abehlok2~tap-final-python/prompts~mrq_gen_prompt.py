from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.chains import LLMCheckerChain
content = ("Fill the role of a question generation engine that generates age and grade appropriate questions for a 4th "
           "grader in New York State, according to the standards set by the New York State Department of Education. "
           "The user will provide the subject, topic, difficulty, number of questions, and number of choices per "
           "question for you to generate. Your questions should be exclusively multiple-response questions.")


mrq_gen_sys_msg = SystemMessage(content=content)

mrq_gen_template = """Generate {numq} example homework questions of {difficulty} difficulty that are appropriate 
                  for a 4th grader in New York State, according to the standards set by the New York State Department 
                  of Education. The subject for these question(s) is: {subject}. The topic for this question is: 
                  {topic}. Each question should have {num_choices} answers to choose from. Please
                  provide the answer to the question in dictionary format. Do not repeat the same question multiple 
                  times in your response."""

input_variables =["numq", "difficulty", "subject", "topic",  "num_choices"]

mrq_gen_prompt_input = HumanMessagePromptTemplate.from_template(template = mrq_gen_template)

mrq_chat_prompt = ChatPromptTemplate.from_messages([mrq_gen_sys_msg, mrq_gen_prompt_input])

