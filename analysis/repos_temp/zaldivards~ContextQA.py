"""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

If the final message aka the follow up input is a gratitude or goodbye message, that MUST be your final answer

Example 1:
Assistant: And that is today's wheather
Human: ok thank you
Standalone question: Thank you

Example 2:
Assistant: And that is today's wheather
Human: ok goodbye
Standalone question: Goodbye


Current conversation:
{chat_history}
Follow Up Input: {question}
Standalone question:""""""
You always need to use the first observation as the final answer:

```
Example 1:
Thought: Do I need to use a tool? Yes
Action: Crawl google for external knowledge
Action Input: Langchain
Observation: This is the result, Langchain is a great framework for LLms...
{ai_prefix}: [Last observation as the answer]
Example 2:
Thought: Do I need to use a tool? Yes
Action: Crawl google for external knowledge
Action Input: Wheater
Observation: This is the whather
{ai_prefix}: [The found wheater]
```

The Thought/Action/Action Input/Observation can repeat only ONCE or answer I don't know:
```
Example 1:
Thought: I now know the final answer
{ai_prefix}: the final answer to the original input question that must be rephrased in an understandable summary
Example 2:
Thought: I don't know the answer
{ai_prefix}: I couldn't find the answer
```

After getting the answer from the tool, your thought MUST be "I got the answer"

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: Your final answer
```""""""
You must use the tools only and only if you are unable to answer with your own training knowledge, otherwise it will be incorrect.

The first observation AFTER using a tool, is your final answer. Use the tool only ONE time:
Obervation: I got the response: [the response]
Thought: Do I need to use a tool? No
{ai_prefix}: [The last observation(the response)]
"""