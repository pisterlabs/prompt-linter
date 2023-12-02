from langchain.prompts.prompt import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, say '''I don't know the answer to this question.'''. Don't try to make up an answer. Write a comprehensive and well formated answer that is readable for a broad and diverse audience of readers. Answer the question as much as possible and only with the provided documents. Don't reference the provided documents but write a extensive answer but also keep the answer concise within a maximum of 8 sentences.
{context}
Question: {question}
Helpful Answer:"""

QA_ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)
