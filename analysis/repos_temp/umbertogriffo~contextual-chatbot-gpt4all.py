"""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the Handbook.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""""""You are an AI assistant specialized in answering questions about the Handbook. 
Given a question and relevant context, provide a conversational answer. If you don't know the answer, respond with, 
'Hmm, I'm not sure.' If the question is unrelated to the Handbook, kindly inform the user that you can only answer 
Handbook-related questions.

Question: {question}
=========
Context: {context}
=========
Answer:""""""Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""