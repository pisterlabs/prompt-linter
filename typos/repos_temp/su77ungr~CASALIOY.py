"""HUMAN:
Answer the question using ONLY the given extracts from (possibly unrelated and irrelevant) documents, not your own knowledge.
If you are unsure of the answer or if it isn't provided in the extracts, answer "Unknown[STOP]".
Conclude your answer with "[STOP]" when you're finished.

Question: {question}

--------------
Here are the extracts:
{context}

--------------
Remark: do not repeat the question !

ASSISTANT:
"""f"""HUMAN:
Answer the question using ONLY the given extracts from a (possibly irrelevant) document, not your own knowledge.
If you are unsure of the answer or if it isn't provided in the extract, answer "Unknown[STOP]".
Conclude your answer with "[STOP]" when you're finished.
Avoid adding any extraneous information.

Question:
-----------------
{{question}}

Extract:
-----------------
{{context}}

ASSISTANT:
"""f"""HUMAN:
Refine the original answer to the question using the new (possibly irrelevant) document extract.
Use ONLY the information from the extract and the previous answer, not your own knowledge.
The extract may not be relevant at all to the question.
Conclude your answer with "[STOP]" when you're finished.
Avoid adding any extraneous information.

Question:
-----------------
{{question}}

Original answer:
-----------------
{{previous_answer}}

New extract:
-----------------
{{context}}

Reminder:
-----------------
If the extract is not relevant or helpful, don't even talk about it. Simply copy the original answer, without adding anything.
Do not copy the question.

ASSISTANT:
""""""HUMAN: Answer the question using ONLY the given context. If you are unsure of the answer, respond with "Unknown[STOP]". Conclude your response with "[STOP]" to indicate the completion of the answer.

Context: {context}

Question: {question}

ASSISTANT:""""""HUMAN: Answer the question using ONLY the given context.
Indicate the end of your answer with "[STOP]" and refrain from adding any additional information beyond that which is provided in the context.

Question: {question}

Context: {context_str}

ASSISTANT:""""""HUMAN: Refine the original answer to the question using the new context.
Use ONLY the information from the context and your previous answer.
If the context is not helpful, use the original answer.
Indicate the end of your answer with "[STOP]" and avoid adding any extraneous information.

Original question: {question}

Existing answer: {existing_answer}

New context: {context_str}

ASSISTANT:"""