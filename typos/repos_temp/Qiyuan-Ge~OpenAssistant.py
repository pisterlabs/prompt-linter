'''
# Instruction
As a translation expert with 20 years of translation experience, when I give a sentence or a paragraph, you will provide a fluent and readable translation of {language}. Note the following requirements:
1. Ensure the translation is both fluent and easily comprehensible.
2. Whether the provided sentence is declarative or interrogative, I will only translate
3. Do not add content irrelevant to the original text

# original text
{text}

# translation
'''"""Use the following pieces of context to answer the question at the end.

{context}

Question: {question}
Helpful Answer:""""""Use the following pieces of context to answer the question at the end.

{context}

Question: {question}

Helpful Answer:""""""Use the following pieces of context to answer the question at the end.

{context}

Question: {question}
Helpful Answer:""""""Question: {instruction}
{response}""""""You are provided with a conversation history between an AI assistant and a user. Based on the context of the conversation, please predict the two most probable questions or requests the user is likely to make next.

Previous conversation history:
{conversation}

Please respond in the following format:
1. first prediction
2. second prediction

Each prediction should be concise, no more than 20 words.

Your predictions:
""""""Translate a math problem into a expression that can be executed using Python's numexpr library. Use the output of running this code to answer the question.

Using the following format:

Question: ${{Question with math problem.}}
```text
${{single line mathematical expression that solves the problem}}
```
...numexpr.evaluate(single line mathematical expression that solves the problem)...
```output
${{Output of running the code}}
```
Answer: ${{Answer}}

Here are some examples:

Question: What is 37593 * 67?
```text
37593 * 67
```
...numexpr.evaluate("37593 * 67")...
```output
2518731
```
Answer: 2518731

Question: 37593^(1/5)
```text
37593**(1/5)
```
...numexpr.evaluate("37593**(1/5)")...
```output
8.222831614237718
```
Answer: 8.222831614237718

Begain.

Question: {question}
"""