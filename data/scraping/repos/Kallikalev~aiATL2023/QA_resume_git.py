from google.cloud import aiplatform
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from vertexai.language_models import TextEmbeddingModel
import os
## DocumentLoaders and TextSplitters are not included here. These are super useful to process/chunk large inputs into prompts

# Two inputs
    # Resume -> assume string
    # repo summaries

def chat_query(resume_string,user_query):

    #Setup VertexAI model for langchain
    llm = VertexAI(model_name="text-bison",max_output_tokens=500,temperature=0.3)


    template = """ You are an tech-recruiter ai assistant answering questions of a candidates resume.
    Resume: {context}

    Give intelligible and useful responses to the following question.

    Question: {question}
    
    Answer:
    """

    ##Setup a prompt
    prompt = PromptTemplate(template=template, input_variables=["context","question"])

    llm_chain = LLMChain(prompt=prompt,llm=llm)



    response = llm_chain.run({"context":resume_string,"question":user_query})
    print(response)

if __name__ == "__main__":
    with open(os.path.join(os.getcwd(),'chad_giga_repo.txt'),'r') as f:
        resume_string = f.read()
    chat_query(resume_string=resume_string,user_query="Where did Chad last work?")

