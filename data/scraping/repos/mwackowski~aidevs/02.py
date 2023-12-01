from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

context = "The Vercel AI SDK is an open-source library designed to help developers build conversational, streaming, and chat user interfaces in JavaScript and TypeScript. The SDK supports React/Next.js, Svelte/SvelteKit, with support for Nuxt/Vue coming soon. To install the SDK, enter the following command in your terminal: npm install ai"

system_template = """
As a {role} who answers the questions ultra-concisely using CONTEXT below 
and nothing more and truthfully says "don't know" when the CONTEXT is not enough to give an answer.

context###{context}###
"""

human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("human", human_template),
    ]
)

formatted_chat_prompt = chat_prompt.format_messages(
    context=context, role="Senior Python Programmer", text="What is Vercel AI?"
)

chat = ChatOpenAI()
content = chat.predict_messages(formatted_chat_prompt)

print(content)
