import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pyairtable import Table
import ast
load_dotenv()

airtable_api_key = os.environ["AIRTABLE_API_KEY"]
base_id = os.environ["AIRTABLE_BASE_ID"]
table_name = os.environ["AIRTABLE_TABLE_NAME"]
open_ai_api_key = os.environ["OPENAI_API_KEY"]



table = Table(airtable_api_key, base_id, table_name)

# we need to wrap the research in double quotes so that it is a valid string
example_research = """According to an article titled "The AI Startup Boom: Analyzing the Latest Trends and Investment Opportunities" on TS2 Space, there are several emerging trends in the AI startup ecosystem:

1. Industry-specific applications: There is an increasing focus on industry-specific applications of AI. While earlier AI startups were primarily focused on developing general-purpose AI technologies, there has been a shift towards more specialized solutions catering to specific industries. Startups are leveraging AI to address unique challenges and pain points within their target industries, making their solutions more valuable and attractive to potential customers and investors.

2. Ethical AI and responsible technology development: There is a growing emphasis on ethical AI and responsible technology development. As AI technologies become more powerful and pervasive, concerns about privacy, security, and fairness have come to the forefront of public discourse. Many AI startups are incorporating ethical considerations into their product development processes and business models, prioritizing the development of AI technologies that are safe, transparent, and aligned with human values.

3. AI-first companies: The increasing maturity of AI technologies has led to a surge in AI startups focused on enhancing existing products and services with AI capabilities. These startups, often referred to as "AI-first" companies, are designing their offerings with AI at the core, rather than simply adding AI features to an existing product. This approach allows AI-first startups to fully harness the power of AI, enabling them to deliver more sophisticated and effective solutions.

4. Investment opportunities: The AI startup boom has attracted significant investment from venture capital firms and tech giants. AI startups have raised billions of dollars in funding, as investors recognize the transformative potential of AI technologies and the growing market demand for AI-powered solutions.

These trends indicate the evolving landscape of the AI startup ecosystem, with a focus on industry-specific applications, ethical considerations, AI-first products, and significant investment opportunities.

Reference:
- Article: [The AI Startup Boom: Analyzing the Latest Trends and Investment Opportunities](https://ts2.space/en/the-ai-startup-boom-analyzing-the-latest-trends-and-investment-opportunities/)"""

def generate_thread(research, subject, airtable_id):

    print("generating thread...")

    # based on research and subject generate a thread using gpt-4, return the thread as a list of strings (each string is a tweet less than 280 characters)
    
    llm = ChatOpenAI(temperature=0, model_name='gpt-4')

    messages = [
    SystemMessage(content="You are an incredibly wise and expert twitter thread writer that generates a high quality and engaging twitter thread based on research provided to you. Your goal is to generate an engaging and thought provoking twitter thread.\n\n% Tone:\n-You should use an active voice and be opinionated\n\n%RESPONSE FORMAT:\n-Do not use any hashtags or mentions.\n-Return your thread as a list of strings (each string is a tweet less than 280 characters)."),
    HumanMessage(content=f"You are a world class twitter thread writer for {subject} and you have been asked to write a thread based on the following research: {research} \n\n Please format your thread as a list of strings (each string is a tweet less than 280 characters)."),
    ]

    response = llm(messages)
    response_content = response.content

    # add to airtable
    print("Adding idea to airtable...")
    thread_data = {
        "Thread": response_content,
    }
    table.update(airtable_id, thread_data)

    # Convert string representation of list to list
    response_content = ast.literal_eval(response_content)
    

    return response_content


# thread = generate_thread(example_research, "AI startups", 0)

# print(thread)

# print("tweet 1", thread[0])
# print("tweet 2", thread[1])
# print("tweet 3", thread[2])

