from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain

llm = LlamaCpp(model_path="./models/vicuna-7b-cot.Q2_K.gguf", n_ctx=4096, n_gpu_layers=43, n_batch=512, verbose=True)

prompt_template_qa = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. 

USER: Read this Content Carefully. {context} 

USER: {question} Answer this question from the context provided.

ASSISTANT:

"""

content = """ 

 1. early childhood care and education: the foundation of learning 
1.1. over 85% of a child’s cumulative brain development occurs prior to the age of 6, indicating the 
critical importance of appropriate care and stimulation of the brain in the early years in order to 
ensure healthy brain development and growth. presently, quality ecce is not available to crores of 
young children, particularly children from socio-economically disadvantaged backgrounds. strong 
investment in ecce has the potential to give all young children such access, enabling them to 
participate and flourish in the educational system throughout their lives. universal provisioning of 
quality early childhood development, care, and education must thus be achieved as soon as possible, 
and no later than 2030, to ensure that all students entering grade 1 are school ready. the middle stage will comprise three years of education, 
building on the pedagogical and curricular style of the preparatory stage, but with the introduction 
of subject teachers for learning and discussion of the more abstract concepts in each subject that 
students will be ready for at this stage across the sciences, mathematics, arts, social sciences, and 
humanities. experiential learning within each subject, and explorations of relations among different 
subjects, will be encouraged and emphasized despite the introduction of more specialized subjects 
and subject teachers. the secondary stage will comprise of four years of multidisciplinary study, 
building on the subject-oriented pedagogical and curricular style of the middle stage, but with greater 
depth, greater critical thinking, greater attention to life aspirations, and greater flexibility and student 
choice of subjects.

"""

question = "how long does it take for brain to develop to 90% ? "

def get_answer_from_llm(content, question):
  PROMPT = PromptTemplate(template=prompt_template_qa, input_variables=["context", "question"])
  llm_chain = LLMChain(prompt=PROMPT, llm=llm)
  print("CONTENT:", content)
  result = llm_chain.run(context=content, question=question)
  print("RESULT:", result)
  return result

get_answer_from_llm(content, question)