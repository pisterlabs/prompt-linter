from langchain.llms import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate

llm = OpenAI()
parser = CommaSeparatedListOutputParser()

recipe_prompt = PromptTemplate(
    template="""
    From the video game Breath of the wild, list the ingredients for {dish}

    {format_instructions}
""",
    input_variables=["dish"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
                               )

query = recipe_prompt.format(dish="crab risotto")
print(query)
response = llm(query)
ingredients = parser.parse(response)
print(sorted(ingredients))