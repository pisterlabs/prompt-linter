from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI

template = """
You are the editor of a newspaper.

Extract the main 5 points from the following article. Return them as a numbered list. Here is the article:\n\n{article}
"""

prompt_template = PromptTemplate(input_variables=["article"], template=template)

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=144)

chain = LLMChain(llm=llm, prompt=prompt_template)

article = """
Ecma aims to develop standards and technical reports to facilitate and standardize the use of information communication technology and consumer electronics; encourage the correct use of standards by influencing the environment in which they are applied; and publish these standards and reports in electronic and printed form. Ecma publications, including standards, can be freely copied by all interested parties without copyright restrictions. The development of standards and technical reports is done in co-operation with the appropriate national, European, and international organizations.

Unlike national standardization bodies, Ecma is a membership-based organization. It takes pride in the resulting "business-like" approach to standards, claimed to lead to better standards in less time, thanks to a less bureaucratic process focused on achieving results by consensus.[citation needed]

Ecma has actively contributed to worldwide standardization in information technology and telecommunications. More than 400 Ecma Standards[2] and 100 Technical Reports[3] have been published, more than 2⁄3 of which have also been adopted as international standards and/or technical reports.

The memberlist of Ecma International is available on its website.[4] Its members include IT companies, IT trade associations, universities, foundations and public institutions.
"""

# points = chain.run(article=article)

# print(points)

simplify_template = """
You are a person who writes provocative tweets.

Condense the following points into a single tweet of less than 145 characters. Here are the points:\n\n{points}
"""

simplify_prompt = PromptTemplate(input_variables=["points"], template=simplify_template)

tweet_llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=144)

simplify_chain = LLMChain(llm=tweet_llm, prompt=simplify_prompt)

simple_chain = SimpleSequentialChain(chains=[chain, simplify_chain], verbose=True)

print(simple_chain.run(article))