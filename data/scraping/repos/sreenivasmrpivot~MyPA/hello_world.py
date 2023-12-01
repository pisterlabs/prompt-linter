from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import LLMChain

# create a new model instance and load the model
llm = CTransformers(model='TheBloke/Llama-2-7b-Chat-GGUF', model_path="llama-2-7b-chat.Q4_K_M.gguf")

# calling llm directly
# response = llm("Who directed The Dark Night?")
# print(response)

# create a new prompt template
template = "Who directed {moviename}?"

prompt = PromptTemplate.from_template(
    input_variables=["moviename"],
    template = template)

prompt.input_variables

# prompt.template

# format the prompt with input variable values
# formtted_prompt = prompt.format(movie_name="The Dark Night")
# formtted_prompt

# define chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# run the chain with input variable values
llm_chain.run("The Dark Night")