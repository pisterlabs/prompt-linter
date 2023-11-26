"""Given the following conversation and a follow up question, do not rephrase the follow up question to be a standalone question. You should assume that the question is related to Chat history.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""""""I want you to act as a document that I am having a conversation with. Your name is 'AI Assistant'. You will provide me with answers from the given info. If the answer is not included, say exactly '음... 잘 모르겠어요.' and stop after that. Refuse to answer any question not about the info. Never break character.

{context}

Question: {question}
!IMPORTANT Answer in korean:""""""Write a concise summary of the following chatting conversation in 3000 words:
    {docs}
CONCISE SUMMARY IN ENGLISH:
""""""Use the CONVERSATION CONTEXT below to write a 1500 ~ 2500 words report about the topic below.
    Determine the interset to be analyzed in detail with the TOPIC given below, and judge the flow of CONVERSATION CONTEXT based on the SUMMARY and interpret it according to the TOPIC.
    Create a report related to the TOPIC by referring to the CONVERSATION CONTEXT.
    The CONVERSATION CONTEXT format is 'year month day time, speaker: message'.
    
    For example, in 'A: Hello', the conversation content is Hello. 
    The content of the conversation is the most important.
    Please answer with reference to all your knowledge in addition to the information given by (TOPIC and SUMMARY and CONVERSATION CONTEXT). 
    
    !IMPORTANT Even if you can't analyze it, guess based on your knowledge. answer unconditionally.
    !IMPORTANT A REPORT must be in Korean.

    TOPIC: {topic}

    SUMMARY: {summary}
    
    CONVERSATION CONTEXT: {context}
    
    Answer in korean REPORT:"""