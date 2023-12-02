from langchain import LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
api_key = "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"
# llm = OpenAI()
llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
embedding_function = OpenAIEmbeddings()

template1 = "Give a summary of this employee's performance review:\n{review}"
prompt1 = ChatPromptTemplate.from_template(template1)
chain_1 = LLMChain(llm=llm,
                     prompt=prompt1,
                     output_key="review_summary")

template2 = "Identify key employee weaknesses in this review summary:\n{review_summary}"
prompt2 = ChatPromptTemplate.from_template(template2)
chain_2 = LLMChain(llm=llm,
                     prompt=prompt2,
                     output_key="weaknesses")

template3 = "Create a personalized plan to help address and fix these weaknesses:\n{weaknesses}"
prompt3 = ChatPromptTemplate.from_template(template3)
chain_3 = LLMChain(llm=llm,
                     prompt=prompt3,
                     output_key="final_plan")

seq_chain = SequentialChain(chains=[chain_1,chain_2,chain_3],
                            input_variables=['review'],
                            output_variables=['review_summary','weaknesses','final_plan'],
                            verbose=True)


employee_review = '''
Employee Information:
Name: Joe Schmo
Position: Software Engineer
Date of Review: July 14, 2023

Strengths:
Joe is a highly skilled software engineer with a deep understanding of programming languages, algorithms, and software development best practices. His technical expertise shines through in his ability to efficiently solve complex problems and deliver high-quality code.

One of Joe's greatest strengths is his collaborative nature. He actively engages with cross-functional teams, contributing valuable insights and seeking input from others. His open-mindedness and willingness to learn from colleagues make him a true team player.

Joe consistently demonstrates initiative and self-motivation. He takes the lead in seeking out new projects and challenges, and his proactive attitude has led to significant improvements in existing processes and systems. His dedication to self-improvement and growth is commendable.

Another notable strength is Joe's adaptability. He has shown great flexibility in handling changing project requirements and learning new technologies. This adaptability allows him to seamlessly transition between different projects and tasks, making him a valuable asset to the team.

Joe's problem-solving skills are exceptional. He approaches issues with a logical mindset and consistently finds effective solutions, often thinking outside the box. His ability to break down complex problems into manageable parts is key to his success in resolving issues efficiently.

Weaknesses:
While Joe possesses numerous strengths, there are a few areas where he could benefit from improvement. One such area is time management. Occasionally, Joe struggles with effectively managing his time, resulting in missed deadlines or the need for additional support to complete tasks on time. Developing better prioritization and time management techniques would greatly enhance his efficiency.

Another area for improvement is Joe's written communication skills. While he communicates well verbally, there have been instances where his written documentation lacked clarity, leading to confusion among team members. Focusing on enhancing his written communication abilities will help him effectively convey ideas and instructions.

Additionally, Joe tends to take on too many responsibilities and hesitates to delegate tasks to others. This can result in an excessive workload and potential burnout. Encouraging him to delegate tasks appropriately will not only alleviate his own workload but also foster a more balanced and productive team environment.
'''

results = seq_chain(employee_review)

print(results['review'])
print(results['review_summary'])
print(results['weaknesses'])
print(results['final_plan'])

