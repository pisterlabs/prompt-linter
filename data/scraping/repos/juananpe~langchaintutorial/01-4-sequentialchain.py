from langchain.prompts import PromptTemplate
from langchain import OpenAI
from langchain.llms import Anthropic
from langchain.chains import SimpleSequentialChain, LLMChain
from dotenv import load_dotenv


load_dotenv()

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write me an outline on {topic}",
)

# llm = Anthropic(temperature=0.5, max_tokens_to_sample=1024)
llm = OpenAI(temperature=0.9, max_tokens=-1)

chain = LLMChain(
    llm=llm, 
    prompt=prompt)

chain = LLMChain(llm=llm, prompt=prompt)

second_prompt = PromptTemplate(        
        input_variables=["outline"], 
        template="""Write a blog article in the format of the given outline
                    Outline:
                    {outline}""",
        )

chain_two = LLMChain(llm=llm, prompt=second_prompt)
overall_chain = SimpleSequentialChain (chains = [chain, chain_two], verbose=True)

# Run the chain specifying only the input variable for the first chain. 
catchphrase = overall_chain.run("Learning SQL")
print(catchphrase)
