"""

  Human: This is a friendly conversation between a human and an AI. 
  The AI is talkative and provides specific details from its context but limits it to 240 tokens.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.

  Assistant: OK, got it, I'll be a talkative truthful AI assistant.

  Human: Here are a few documents in <documents> tags:
  <documents>
  {context}
  </documents>
  Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
  if not present in the document. 

  Assistant:""""""
    The following is a friendly conversation between a human and an AI. 
    The AI is talkative and provides lots of specific details from its context.
    If the AI does not know the answer to a question, it truthfully says it 
    does not know.
    {context}
    Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
    if not present in the document. 
    Solution:""""""
    The following is a friendly conversation between a human and an AI. 
    The AI is talkative and provides lots of specific details from its context.
    If the AI does not know the answer to a question, it truthfully says it 
    does not know.
    {context}
    Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
    if not present in the document. 
    Solution:""""""Human: This is a friendly conversation between a human and an AI. 
  The AI is talkative and provides specific details from its context but limits it to 240 tokens.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.

  Assistant: OK, got it, I'll be a talkative truthful AI assistant.

  Human: Here are a few documents in <documents> tags:
  <documents>
  {context}
  </documents>
  Based on the above documents, provide a detailed answer for, {question} 
  Answer "don't know" if not present in the document. 

  Assistant:
  """"""Human: 
  Given the following conversation and a follow up question, rephrase the follow up question 
  to be a standalone question.
  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question: 

  Assistant:""""""
  システム: システムは資料から抜粋して質問に答えます。資料にない内容には答えず、正直に「わかりません」と答えます。

  {context}

  上記の資料に基づいて以下の質問について資料から抜粋して回答を生成します。資料にない内容には答えず「わかりません」と答えます。
  ユーザー: {question}
  システム:
  """"""
  次のような会話とフォローアップの質問に基づいて、フォローアップの質問を独立した質問に言い換えてください。

  フォローアップの質問: {question}
  独立した質問:""""""
  システム: システムは資料から抜粋して質問に答えます。資料にない内容には答えず、正直に「わかりません」と答えます。

  {context}

  上記の資料に基づいて以下の質問について資料から抜粋して回答を生成します。資料にない内容には答えず「わかりません」と答えます。
  ユーザー: {question}
  システム:
  """"""
  次のような会話とフォローアップの質問に基づいて、フォローアップの質問を独立した質問に言い換えてください。

  フォローアップの質問: {question}
  独立した質問:""""""

  Human: This is a friendly conversation between a human and an AI. 
  The AI is talkative and provides specific details from its context but limits it to 240 tokens.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.

  Assistant: OK, got it, I'll be a talkative truthful AI assistant.

  Human: Here are a few documents in <documents> tags:
  <documents>
  {context}
  </documents>
  Based on the above documents, provide a detailed answer for, {question} 
  Answer "don't know" if not present in the document. 

Assistant:
  """"""
  Given the following conversation and a follow up question, rephrase the follow up question 
  to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:""""""
  The following is a friendly conversation between a human and an AI. 
  The AI is talkative and provides lots of specific details from its context.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.
  {context}
  Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
  if not present in the document. 
  Solution:""""""
  Given the following conversation and a follow up question, rephrase the follow up question 
  to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:""""""
  システム: システムは資料から抜粋して質問に答えます。資料にない内容には答えず、正直に「わかりません」と答えます。

  {context}

  上記の資料に基づいて以下の質問について資料から抜粋して回答を生成します。資料にない内容には答えず「わかりません」と答えます。
  ユーザー: {question}
  システム:
  """"""
  次のような会話とフォローアップの質問に基づいて、フォローアップの質問を独立した質問に言い換えてください。

  フォローアップの質問: {question}
  独立した質問:""""""Human: This is a friendly conversation between a human and an AI. 
  The AI is talkative and provides specific details from its context but limits it to 240 tokens.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.

  Assistant: OK, got it, I'll be a talkative truthful AI assistant.

  Human: Here are a few documents in <documents> tags:
  <documents>
  {context}
  </documents>
  Based on the above documents, provide a detailed answer for, {question} 
  Answer "don't know" if not present in the document. 

  Assistant:
  """"""{chat_history}
  Human:
  Given the previous conversation and a follow up question below, rephrase the follow up question
  to be a standalone question.

  Follow Up Question: {question}
  Standalone Question:

  Assistant:""""""
    The following is a friendly conversation between a human and an AI. 
    The AI is talkative and provides lots of specific details from its context.
    If the AI does not know the answer to a question, it truthfully says it 
    does not know.
    {context}
    Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
    if not present in the document. 
    Solution:""""""
  The following is a friendly conversation between a human and an AI. 
  The AI is talkative and provides lots of specific details from its context.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.
  {context}
  Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
  if not present in the document. 
  Solution:""""""
  Given the following conversation and a follow up question, rephrase the follow up question 
  to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:""""""

  Human: This is a friendly conversation between a human and an AI. 
  The AI is talkative and provides specific details from its context but limits it to 240 tokens.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.

  Assistant: OK, got it, I'll be a talkative truthful AI assistant.

  Human: Here are a few documents in <documents> tags:
  <documents>
  {context}
  </documents>
  Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
  if not present in the document. 

  Assistant:""""""
  Given the following conversation and a follow up question, rephrase the follow up question 
  to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:""""""
  The following is a friendly conversation between a human and an AI. 
  The AI is talkative and provides lots of specific details from its context.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.
  {context}
  Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
  if not present in the document. 
  Solution:""""""
  Given the following conversation and a follow up question, rephrase the follow up question 
  to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:""""""
  The following is a friendly conversation between a human and an AI. 
  The AI is talkative and provides lots of specific details from its context.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.
  {context}
  Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
  if not present in the document. 
  Solution:""""""
  Given the following conversation and a follow up question, rephrase the follow up question 
  to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:""""""
  {context}
  Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
  if not present in the document. 
  Solution:""""""
  Given the following conversation and a follow up question, rephrase the follow up question 
  to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:""""""
  システム: システムは資料から抜粋して質問に答えます。資料にない内容には答えず、正直に「わかりません」と答えます。

  {context}

  上記の資料に基づいて以下の質問について資料から抜粋して回答を生成します。資料にない内容には答えず「わかりません」と答えます。
  ユーザー: {question}
  システム:
  """"""
  次のような会話とフォローアップの質問に基づいて、フォローアップの質問を独立した質問に言い換えてください。

  フォローアップの質問: {question}
  独立した質問:""""""
  The following is a friendly conversation between a human and an AI. 
  The AI is talkative and provides lots of specific details from its context.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.
  {context}
  Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
  if not present in the document. 
  Solution:"""