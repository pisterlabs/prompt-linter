from langchain.llms import OpenAI
from langchain import PromptTemplate

template = """
I want you to act as a naming consultant for new companies.

Here are some examples of good company names:

- search engine, Google
- social media, Facebook
- video sharing, YouTube

The name should be short, catchy and easy to remember.

What is a good name for a company that makes {product}?
"""

prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)

llm = OpenAI(temperature=0.9)
text = prompt.format(product="deep learning")
print(llm(text))
