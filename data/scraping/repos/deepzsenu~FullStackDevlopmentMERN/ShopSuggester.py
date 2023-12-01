import openai
from langchain.llms import OpenAI
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

from secret import API_KEY

llm = OpenAI(openai_api_key=API_KEY, temperature=0.5)

def shop_suggestions(name):
    template = """
        Suggest me a Store name
        {name}
        in one word
        """
    name_prompt = PromptTemplate(template=template, input_variables=['name'])
    name_chain = LLMChain(llm=llm, prompt=name_prompt, output_key='store_name')

    template = """
        What are the names of the products that are kept in the store
        {store_name}
        in bullet points
        """
    products_prompt = PromptTemplate(template=template, input_variables=['store_name'])
    products_chain = LLMChain(llm=llm, prompt=products_prompt, output_key='products')

    final_model = SequentialChain(
        chains = [name_chain, products_chain],
        input_variables=["name"],
        output_variables=["store_name", 'products']

    )

    text = final_model({'name':name})

    return text


