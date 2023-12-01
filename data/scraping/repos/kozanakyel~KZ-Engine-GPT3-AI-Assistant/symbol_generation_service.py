from langchain import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv
import os
from langchain import OpenAI

load_dotenv()

openai_api_key = os.getenv('T_OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

openai = OpenAI(
    #model_name='text-davinci-003',
    model_name='text-davinci-003',
    temperature=0
)

examples = [
    {
        "query": "I want to get trading advice about the ethereum.",
        "answer": "ETH-USD"
    },
    {
        "query": "Can you give trading advice about the solana?",
        "answer": "SOL-USD"
    }
]

# create a example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """Answer the question based on the context below.
You are an AI assistant who knows about cryptocurrency and can find the Yahoo Finance symbol for any given cryptocurrency. 

Context: When given a request for trading advice on a specific cryptocurrency, 
your task is to find and provide the corresponding Yahoo Finance symbol for that cryptocurrency. 
If the query dont have any cryptocurency names you should return BTC-USD.
The symbol usually combines the cryptocurrency's ticker symbol and 'USD'. Here are some examples:
"""

# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

class SymboGenerationPromptService:
    
    @staticmethod
    def get_symbol(query: str):
        return openai(
            few_shot_prompt_template.format(
                query= query
            )
        )
        
if __name__ == '__main__':
    res = SymboGenerationPromptService.get_symbol("I want to get a trading advice about the?")
    print(res)