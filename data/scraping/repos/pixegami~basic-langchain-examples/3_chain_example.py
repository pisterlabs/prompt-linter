from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chat_models.openai import ChatOpenAI

prompt_1 = """
Come up with a short plot synopsis summary (2-3 lines) for a {genre} movie.
"""

prompt_2 = """
Suggest 5 movie title ideas for this movie: \n\n {synopsis}
"""

llm = ChatOpenAI()

p1_template = PromptTemplate.from_template(prompt_1)
chain_1 = LLMChain(llm=llm, prompt=p1_template, output_key="synopsis")

p2_template = PromptTemplate.from_template(prompt_2)
chain_2 = LLMChain(llm=llm, prompt=p2_template, output_key="titles")

complex_chain = SequentialChain(
    chains=[chain_1, chain_2],
    input_variables=["genre"],
    output_variables=["synopsis", "titles"],
    verbose=True,
)
output = complex_chain({"genre": "comedy"})
print(f"Output: {output}")
