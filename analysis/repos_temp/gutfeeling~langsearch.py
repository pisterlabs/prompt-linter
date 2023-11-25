"""You are an expert in the Python programming language and you like to provide helpful answers to questions. Please answer the following question.
Question: {QUESTION}
Answer:""""""Answer the question as truthfully as possible using the following context, and if the answer is not contained in the context, say "I don't know."
Context:
{context}

Question: {question}
Answer, according to the supplied context: """"""Answer the question as truthfully as possible using the following context, and if the answer is not contained in the context, say "I don't know."
Context:
{context}

Question: {question}
Answer, according to the supplied context: """"""You are trying to find links that might contain the answer to the question: {question}

You have a few links, but you can't view all the information contained under the link. You only have access to a concise and incomplete summary of the information contained in those links. Therefore, the summaries may not contain the answer to the question directly. The links themselves contain a lot more information than the summary. You need to decide which links to investigate further, i.e view their full content.

{context}

For which links would you fetch the full content to see if they contain the answer to the following question: {question}

Remember, the summaries may not contain the answer to the question directly, because they are incomplete. The links themselves contain a lot more information than the summary.

Please provide a list of all those links to investigate further.

List of links:
""""""{text}

Tl;dr
"""