from langchain import PromptTemplate
from reasoning import openai, davinci1

pt = PromptTemplate(
    input_variables=["query"],
    template="""{query}
When you reply, first find exact quotes in the FAQ relevant to the user's question and write them down word for word inside <thinking></thinking> XML tags.  This is a space for you to write down relevant content and will not be shown to the user.  Once you are done extracting relevant quotes, answer the question.  Put your answer to the user inside <answer></answer> XML tags."""
)

pt_verbose = PromptTemplate(
    input_variables=["query"],
    template="""{query}
When you reply, first find exact quotes in the FAQ relevant to the user's question and write them down word for word inside <thinking></thinking> XML tags.  This is a space for you to write down relevant content explicitly.  Once you are done extracting relevant quotes, answer the question.  Put your answer to the user inside <answer></answer> XML tags."""
)

pt_verbose2 = PromptTemplate(
    input_variables=["query"],
    template="""Answer the question.
{query}
Let’s write word by word and think step by step until we arrive to an answer.
All the steps of the reasoning should be written inside <thinking></thinking> XML tags. Format the thoughts a short clear statements.
While writing, you can use the FAQ to find relevant quotes.  Write them down word for word inside <thinking></thinking> XML tags.
While writing, you can use information from <thinking></thinking>.
Put the answer inside <answer></answer> XML tags.
Therefore:""")

question = "I have 4 apples and I give you two of my apples. After this I buy one apple and eat it. How many apples do I have left?"

print("openai silent")
print("question:", question)
print(openai(pt.format(query=question)))
print("*"*80)
print("openai verbose")
print("question:", question)
print(davinci1(pt_verbose.format(query=question)))
print("*"*80)
print("openai verbose 2")
print("question:", question)
print(openai(pt_verbose2.format(query=question)))
print("*"*80)
print("davinci silent")
print("question:", question)
print(davinci1(pt.format(query=question)))
print("*"*80)
print("davinci verbose")
print("question:", question)
print(davinci1(pt_verbose.format(query=question)))
print("*"*80)
print("davinci verbose 2")
print("question:", question)
print(davinci1(pt_verbose2.format(query=question)))
print("*"*80)
