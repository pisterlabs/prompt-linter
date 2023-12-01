#!/usr/bin/env python
# coding: utf-8

# # Generating Answers

# In[ ]:


question = "Are side projects important when you are starting to learn about AI?"


# In[2]:


text = """
The rapid rise of AI has led to a rapid rise in AI jobs, and many people are building exciting careers in this field. A career is a decades-long journey, and the path is not always straightforward. Over many years, I’ve been privileged to see thousands of students as well as engineers in companies large and small navigate careers in AI. In this and the next few letters, I’d like to share a few thoughts that might be useful in charting your own course.

Three key steps of career growth are learning (to gain technical and other skills), working on projects (to deepen skills, build a portfolio, and create impact) and searching for a job. These steps stack on top of each other:

Initially, you focus on gaining foundational technical skills.
After having gained foundational skills, you lean into project work. During this period, you’ll probably keep learning.
Later, you might occasionally carry out a job search. Throughout this process, you’ll probably continue to learn and work on meaningful projects.
These phases apply in a wide range of professions, but AI involves unique elements. For example:

AI is nascent, and many technologies are still evolving. While the foundations of machine learning and deep learning are maturing — and coursework is an efficient way to master them — beyond these foundations, keeping up-to-date with changing technology is more important in AI than fields that are more mature.
Project work often means working with stakeholders who lack expertise in AI. This can make it challenging to find a suitable project, estimate the project’s timeline and return on investment, and set expectations. In addition, the highly iterative nature of AI projects leads to special challenges in project management: How can you come up with a plan for building a system when you don’t know in advance how long it will take to achieve the target accuracy? Even after the system has hit the target, further iteration may be necessary to address post-deployment drift.
While searching for a job in AI can be similar to searching for a job in other sectors, there are some differences. Many companies are still trying to figure out which AI skills they need and how to hire people who have them. Things you’ve worked on may be significantly different than anything your interviewer has seen, and you’re more likely to have to educate potential employers about some elements of your work.
Throughout these steps, a supportive community is a big help. Having a group of friends and allies who can help you — and whom you strive to help — makes the path easier. This is true whether you’re taking your first steps or you’ve been on the journey for years.

I’m excited to work with all of you to grow the global AI community, and that includes helping everyone in our community develop their careers. I’ll dive more deeply into these topics in the next few weeks.

Last week, I wrote about key steps for building a career in AI: learning technical skills, doing project work, and searching for a job, all of which is supported by being part of a community. In this letter, I’d like to dive more deeply into the first step.

More papers have been published on AI than any person can read in a lifetime. So, in your efforts to learn, it’s critical to prioritize topic selection. I believe the most important topics for a technical career in machine learning are:

Foundational machine learning skills. For example, it’s important to understand models such as linear regression, logistic regression, neural networks, decision trees, clustering, and anomaly detection. Beyond specific models, it’s even more important to understand the core concepts behind how and why machine learning works, such as bias/variance, cost functions, regularization, optimization algorithms, and error analysis.
Deep learning. This has become such a large fraction of machine learning that it’s hard to excel in the field without some understanding of it! It’s valuable to know the basics of neural networks, practical skills for making them work (such as hyperparameter tuning), convolutional networks, sequence models, and transformers.
Math relevant to machine learning. Key areas include linear algebra (vectors, matrices, and various manipulations of them) as well as probability and statistics (including discrete and continuous probability, standard probability distributions, basic rules such as independence and Bayes rule, and hypothesis testing). In addition, exploratory data analysis (EDA) — using visualizations and other methods to systematically explore a dataset — is an underrated skill. I’ve found EDA particularly useful in data-centric AI development, where analyzing errors and gaining insights can really help drive progress! Finally, a basic intuitive understanding of calculus will also help. In a previous letter, I described how the math needed to do machine learning well has been changing. For instance, although some tasks require calculus, improved automatic differentiation software makes it possible to invent and implement new neural network architectures without doing any calculus. This was almost impossible a decade ago.
Software development. While you can get a job and make huge contributions with only machine learning modeling skills, your job opportunities will increase if you can also write good software to implement complex AI systems. These skills include programming fundamentals, data structures (especially those that relate to machine learning, such as data frames), algorithms (including those related to databases and data manipulation), software design, familiarity with Python, and familiarity with key libraries such as TensorFlow or PyTorch, and scikit-learn.
This is a lot to learn! Even after you master everything in this list, I hope you’ll keep learning and continue to deepen your technical knowledge. I’ve known many machine learning engineers who benefitted from deeper skills in an application area such as natural language processing or computer vision, or in a technology area such as probabilistic graphical models or building scalable software systems.

How do you gain these skills? There’s a lot of good content on the internet, and in theory reading dozens of web pages could work. But when the goal is deep understanding, reading disjointed web pages is inefficient because they tend to repeat each other, use inconsistent terminology (which slows you down), vary in quality, and leave gaps. That’s why a good course — in which a body of material has been organized into a coherent and logical form — is often the most time-efficient way to master a meaningful body of knowledge. When you’ve absorbed the knowledge available in courses, you can switch over to research papers and other resources.

Finally, keep in mind that no one can cram everything they need to know over a weekend or even a month. Everyone I know who’s great at machine learning is a lifelong learner. In fact, given how quickly our field is changing, there’s little choice but to keep learning if you want to keep up. How can you maintain a steady pace of learning for years? I’ve written about the value of habits. If you cultivate the habit of learning a little bit every week, you can make significant progress with what feels like less effort.

In the last two letters, I wrote about developing a career in AI and shared tips for gaining technical skills. This time, I’d like to discuss an important step in building a career: project work.

It goes without saying that we should only work on projects that are responsible and ethical, and that benefit people. But those limits leave a large variety to choose from. I wrote previously about how to identify and scope AI projects. This and next week’s letter have a different emphasis: picking and executing projects with an eye toward career development.

A fruitful career will include many projects, hopefully growing in scope, complexity, and impact over time. Thus, it is fine to start small. Use early projects to learn and gradually step up to bigger projects as your skills grow.

When you’re starting out, don’t expect others to hand great ideas or resources to you on a platter. Many people start by working on small projects in their spare time. With initial successes — even small ones — under your belt, your growing skills increase your ability to come up with better ideas, and it becomes easier to persuade others to help you step up to bigger projects.

What if you don’t have any project ideas? Here are a few ways to generate them:

Join existing projects. If you find someone else with an idea, ask to join their project.
Keep reading and talking to people. I come up with new ideas whenever I spend a lot of time reading, taking courses, or talking with domain experts. I’m confident that you will, too.
Focus on an application area. Many researchers are trying to advance basic AI technology — say, by inventing the next generation of transformers or further scaling up language models — so, while this is an exciting direction, it is hard. But the variety of applications to which machine learning has not yet been applied is vast! I’m fortunate to have been able to apply neural networks to everything from autonomous helicopter flight to online advertising, partly because I jumped in when relatively few people were working on those applications. If your company or school cares about a particular application, explore the possibilities for machine learning. That can give you a first look at a potentially creative application — one where you can do unique work — that no one else has done yet.
Develop a side hustle. Even if you have a full-time job, a fun project that may or may not develop into something bigger can stir the creative juices and strengthen bonds with collaborators. When I was a full-time professor, working on online education wasn’t part of my “job” (which was doing research and teaching classes). It was a fun hobby that I often worked on out of passion for education. My early experiences recording videos at home helped me later in working on online education in a more substantive way. Silicon Valley abounds with stories of startups that started as side projects. So long as it doesn’t create a conflict with your employer, these projects can be a stepping stone to something significant.
Given a few project ideas, which one should you jump into? Here’s a quick checklist of factors to consider:

Will the project help you grow technically? Ideally, it should be challenging enough to stretch your skills but not so hard that you have little chance of success. This will put you on a path toward mastering ever-greater technical complexity.
Do you have good teammates to work with? If not, are there people you can discuss things with? We learn a lot from the people around us, and good collaborators will have a huge impact on your growth.
Can it be a stepping stone? If the project is successful, will its technical complexity and/or business impact make it a meaningful stepping stone to larger projects? (If the project is bigger than those you’ve worked on before, there’s a good chance it could be such a stepping stone.)
Finally, avoid analysis paralysis. It doesn’t make sense to spend a month deciding whether to work on a project that would take a week to complete. You'll work on multiple projects over the course of your career, so you’ll have ample opportunity to refine your thinking on what’s worthwhile. Given the huge number of possible AI projects, rather than the conventional “ready, aim, fire” approach, you can accelerate your progress with “ready, fire, aim.”

"""


# ## Setup
# 
# Load needed API keys and relevant Python libaries.

# In[3]:


import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# In[ ]:


import cohere

import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ## Chunking

# In[4]:


# Split into a list of paragraphs
texts = text.split('\n\n')

# Clean up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts if t])


# In[6]:


texts[:3]


# ## Embeddings

# In[7]:


co = cohere.Client(os.environ['COHERE_API_KEY'])

# Get the embeddings
response = co.embed(
    texts=texts.tolist(),
).embeddings


# ## Build a search index

# In[8]:


from annoy import AnnoyIndex
import numpy as np
import pandas as pd


# In[9]:


# Check the dimensions of the embeddings
embeds = np.array(response)

# Create the search index, pass the size of embedding
search_index = AnnoyIndex(embeds.shape[1], 'angular')
# Add all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10) # 10 trees
search_index.save('test.ann')


# ## Searching Articles

# In[10]:


def search_andrews_article(query):
    # Get the query's embedding
    query_embed = co.embed(texts=[query]).embeddings
    
    # Retrieve the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0],
                                                    10,
                                                  include_distances=True)

    search_results = texts[similar_item_ids[0]]
    
    return search_results


# In[11]:


results = search_andrews_article(
    "Are side projects a good idea when trying to build a career in AI?"
)

print(results[0])


# ## Generating Answers

# In[43]:


def ask_andrews_article(question, num_generations=1):
    
    # Search the text archive
    results = search_andrews_article(question)

    # Get the top result
    context = results[0]

    # Prepare the prompt
    prompt = f"""
    Excerpt from the article titled "How to Build a Career in AI" 
    by Andrew Ng: 
    {context}
    Question: {question}
    
    Extract the answer of the question from the text provided. 
    If the text doesn't contain the answer, 
    reply that the answer is not available."""

    prediction = co.generate(
        prompt=prompt,
        max_tokens=70,
        model="command-nightly",
        temperature=0.5,
        num_generations=num_generations
    )

    return prediction.generations


# In[22]:


results = ask_andrews_article(
    "Are side projects a good idea when trying to build a career in AI?",

)

print(results[0])


# In[25]:


results = ask_andrews_article(
    "Are side projects a good idea when trying to build a career in AI?",
    num_generations=3
)

for gen in results:
    print(gen)
    print('--')


# In[44]:


results = ask_andrews_article(
    "What is the most viewed televised event?",
    num_generations=5
)


# In[45]:


for gen in results:
    print(gen)
    print('--')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Congratulations on finishing the course!
# 
# To start building with the Cohere LLMs, get your API key by registering [here](https://dashboard.cohere.ai/welcome/register?utm_source=partner&utm_medium=website&utm_campaign=DeeplearningAI). 
# 
# Learn more about LLMs at Cohere’s [LLM.University](https://LLM.University).