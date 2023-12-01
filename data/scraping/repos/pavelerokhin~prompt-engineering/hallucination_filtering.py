from hallucinations import openai
from langchain import PromptTemplate

pt = PromptTemplate(
    input_variables=["question"],
    template="""You are physicist. Your job is to answer questions about physics.
If you don't know the answer, say "I don't know".

User: {question}
Physicist: """
)

pt_hallucination_filter = PromptTemplate(
    input_variables=["question"],
    template="""You are physicist. Your job is to answer questions about physics.
Answer using only real physics knowledge and reliable sources, cite those sources.
If you don't know the answer, say "I don't know".

User: {question}
Physicist: """
)

print(openai(pt.format(question="4th law of thermodynamics")))
print(openai(pt_hallucination_filter.format(question="4th law of thermodynamics")))
# both results were good enough with gpt-3.5-turbo model
# and they unsatisfactory with davinci model
