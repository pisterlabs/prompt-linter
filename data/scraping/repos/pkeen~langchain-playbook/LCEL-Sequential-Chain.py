import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import StrOutputParser

llm = ChatOpenAI()

employee_review = """

**Company:** ABC Corporation  
**Employee:** John Smith  
**Position:** Senior Sales Manager  
**Review Period:** January 1, 2023, to December 31, 2023

**Overall Performance Rating:** 4.5 out of 5

**Strengths:**

1. **Sales Leadership:** John consistently demonstrated strong leadership in the sales department. His ability to motivate his team and set challenging yet achievable targets was a key factor in the department's success this year.

2. **Customer Relationship Management:** John has excelled in building and maintaining strong relationships with key clients. His personalized approach and deep understanding of their needs have resulted in increased customer satisfaction and repeat business.

3. **Adaptability:** In a rapidly changing market, John has shown a remarkable ability to adapt to new strategies and technologies. He played a pivotal role in implementing our new CRM system, which significantly improved sales efficiency.

4. **Team Collaboration:** John fosters a collaborative work environment. He actively encourages team members to share ideas and best practices, which has led to a more cohesive and productive sales team.

5. **Communication Skills:** John's communication skills are excellent. He regularly communicates departmental updates and performance metrics to senior management, making it easier for us to make informed decisions.

**Areas for Improvement:**

1. **Time Management:** While John's dedication to his role is commendable, he sometimes struggles with time management. There were instances where he missed deadlines, causing minor disruptions in project timelines.

2. **Delegation:** John tends to take on a heavy workload himself, which can lead to burnout. Encouraging him to delegate more effectively could help distribute the workload and improve overall team efficiency.

3. **Feedback Reception:** John sometimes struggles with receiving constructive feedback. Encouraging him to be more open to feedback and using it as a tool for personal growth will benefit both him and the team.

4. **Technology Proficiency:** While John adapted well to the new CRM system, there is room for improvement in his overall technology proficiency. Additional training in this area could further enhance his efficiency.

**Summary:**

John Smith has been a valuable asset to ABC Corporation during the review period. His strengths in sales leadership, customer relationship management, adaptability, team collaboration, and communication skills have contributed significantly to the company's success. However, addressing the identified areas for improvement, such as time management, delegation, feedback reception, and technology proficiency, will help John continue to excel and contribute even more effectively to our organization's growth.

We look forward to seeing John's continued development and contributions in the coming year.

---

Please note that this is a fictional example and should not be used for any actual employee review without appropriate modifications and considerations specific to your organization's policies and procedures.
"""


template1 = "Give a summary of this employees performance review: \n {review}"
prompt1 = ChatPromptTemplate.from_template(template=template1)

template2 = "Identify key employee weaknesses in this review summary: \n {review_summary}"
prompt2 = ChatPromptTemplate.from_template(template=template2)

template3 = "Create a personalized plan to help address and fix these weaknesses \n {weaknesses}"
prompt3 = ChatPromptTemplate.from_template(template=template3)

chain = prompt1 | llm | {"review_summary": StrOutputParser()} | prompt2 | llm | {"weaknesses": StrOutputParser()} | prompt3 | llm

# result = chain.invoke({'review': employee_review})

# print(result.content)
for s in chain.stream({'review': employee_review}):
    print(s.content, end="", flush=True)