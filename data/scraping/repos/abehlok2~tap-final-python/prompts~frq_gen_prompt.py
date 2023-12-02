from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.chains import LLMCheckerChain
content = ("Fill the role of a question generation engine that generates age and grade appropriate questions for a 4th "
           "grader in New York State, according to the standards set by the New York State Department of Education. "
           "The user will provide the subject, topic, difficulty, and number of questions for you to generate. Your"
           "questions should be exclusively free-response questions.")


frq_gen_sys_msg = SystemMessage(content=content)
frq_gen_template = """Generate {numq} example homework questions of {difficulty} difficulty that are appropriate
                  for a 4th grader in New York State, according to the standards set by the New York State Department 
                  of Education. The subject for this homework question is {subject}. The topic for this question is 
                  {topic}. Please provide the answer to the question in dictionary format. Do not repeat the same 
                  question multiple times in your response."""


input_variables =["numq", "difficulty", "subject", "topic"]

frq_gen_prompt_input = HumanMessagePromptTemplate.from_template(template = frq_gen_template)

frq_chat_prompt = ChatPromptTemplate.from_messages([frq_gen_sys_msg, frq_gen_prompt_input])

