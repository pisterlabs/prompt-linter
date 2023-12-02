#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import pandas as pd
import time
from resume_parser import resumeparse
import cohere
import numpy as np
from annoy import AnnoyIndex
import umap.umap_ as umap
import altair as alt

ls_final = []
file = 'Resume.pdf'
data = resumeparse.read_file(file)
ls_jobs = data['designition']
for i in ls_jobs:
    if len(i.split()) == 2:
        ls_final.append(i)
ls_final

for val in ls_final:
    print(val.split()[0])
    print(val.split()[1])


def getInput():
    keywords = []
    while (True):
        word = input("Enter a keyword. Enter \"done\" to finish.")
        if word.lower() == "done":
            break
        keywords.append(word)
    jt = input("full time, permanent, contract, apprenticeship, intern, part-time, temporary, casual")
    searchByKeyword(keywords, jt)


# In[29]:


def searchByKeyword(title1, title2, i, jobList):
    search = "https://ca.indeed.com/jobs?q=" + title1 + "%20" + title2 + "&start=" + str(i)

    session = requests.Session()
    response = session.get(search, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36"})
    soup = BeautifulSoup(response.text, 'html.parser')
    print(search)

    # array of dictionary

    jobList = extract_info(soup)
    links = extract_link(soup)

    for i in range(0, len(links)):
        job_src = requests.get(links[i]).text
        job_soup = BeautifulSoup(job_src, 'lxml')
        salary, job_type, desc = extract_details(job_soup)
        jobList[i]['link'] = links[i]
        jobList[i]['salary'] = salary
        jobList[i]['job_type'] = job_type
        jobList[i]['description'] = desc
    return jobList


def extract_link(soup):
    links = []
    part = soup.find('div', id='mosaic-provider-jobcards')
    for elem in part.find_all('a', class_='tapItem'):
        link = elem.get('href')
        link = "https://ca.indeed.com" + link
        links.append(link)
    return (links)


def extract_info(soup):
    jobs = []
    job_titles = []
    company_names = []
    locations = []
    for elem in soup.find_all('h2', class_='jobTitle'):
        spans = elem.find_all('span')
        for s in spans:
            if s.has_attr('title'):
                title = s.get_text()
        job_titles.append(title)

    for elem in soup.find_all('span', class_='companyName'):
        name = elem.get_text()
        company_names.append(name)

    for elem in soup.find_all('div', class_='companyLocation'):
        loc = elem.get_text()
        if loc.find(" in ") != -1:
            idx = loc.index(" in ") + 4
            loc = loc[idx:]
        locations.append(loc)

    for a in range(0, len(job_titles)):
        dic = {}
        dic['title'] = job_titles[a]
        dic['company'] = company_names[a]
        dic['location'] = locations[a]
        jobs.append(dic)

    return jobs


def extract_details(soup):
    container = soup.find('div', id='salaryInfoAndJobType')
    desc_cont = soup.find('div', id='jobDescriptionText')
    if desc_cont:
        description = desc_cont.get_text()
    salary = ""
    job_type = ""
    if container:
        salary_cont = container.find("span", recursive=False, class_='attribute_snippet')
        job_type_cont = container.find("span", recursive=False, class_='jobsearch-JobMetadataHeader-item')
        if salary_cont:
            salary = salary_cont.get_text()
        if job_type_cont:
            job_type = job_type_cont.get_text()
    return salary, job_type, description


final_list = []

for val in ls_final:
    for i in range(0, 40, 10):
        jobList = []
        jobList = searchByKeyword(val.split()[0], val.split()[1], i, jobList)
        final_list.append(jobList)


dataTitle = []
dataCompany = []
dataSalary = []
dataLocation = []
dataDescription = []
dataType = []
datalink = []
for val in final_list:
    for element in val:
        dataTitle.append(element['title'])
        dataCompany.append(element['company'])
        dataSalary.append(element['salary'])
        dataType.append(element['job_type'])
        dataLocation.append(element['location'])
        dataDescription.append(element['description'])
        datalink.append(element['link'])

df = pd.DataFrame({'title': dataTitle, 'company': dataCompany, 'salary': dataSalary, 'location': dataLocation,
                   'description': dataDescription, 'job type': dataType, 'link': datalink})
df = df.iloc[1:, :]

# In[31]:


len(df)

# In[32]:


df.head()

# In[33]:


string = ' '.join(data['skills'])
string

# In[34]:


co = cohere.Client('T2E1lsDykK0BnbiPJDHXx5OqxWIcb6Fu1tr13Bny')

# In[35]:


embeds = co.embed(texts=list(df['description']), model="large", truncate="LEFT").embeddings

# In[36]:


embeds = np.array(embeds)
embeds.shape

# In[37]:


# Create the search index, pass the size of embedding
search_index = AnnoyIndex(embeds.shape[1], 'angular')
# Add all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10)  # 10 trees
search_index.save('test.ann')

# In[39]:


query = string

# Get the query's embedding
query_embed = co.embed(texts=[query],
                       model="medium",
                       truncate="LEFT").embeddings

# Retrieve the nearest neighbors
similar_item_ids = search_index.get_nns_by_vector(query_embed[0], 10,
                                                  include_distances=True)
# Format the results
results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['description'],
                             'distance': similar_item_ids[1]})
print(f"Query:'{query}'\nNearest neighbors:")
results

# In[26]:


results['texts'].iloc[0]

# In[10]:


for file in files:
    file.delete()

# ### NOTES

# #### We should now start connecting everything together, so we have the website and pdf storage ready, so once the user uploads the resume, we use some post requests commands and figure out a way to call this python notebook, with the input being the directory to the pdf. Once the model runs and produces the dataframe, we return the 10 best jobs, and return those LINKS, and then pass that back into the website, where we post it for them

# In[40]:


reducer = umap.UMAP(n_neighbors=20)
umap_embeds = reducer.fit_transform(embeds)
# Prepare the data to plot and interactive visualization
# using Altair
df_explore = pd.DataFrame(data={'text': df['description']})
df_explore['x'] = umap_embeds[:, 0]
df_explore['y'] = umap_embeds[:, 1]

# Plot
chart = alt.Chart(df_explore).mark_circle(size=60).encode(
    x=  # 'x',
    alt.X('x',
          scale=alt.Scale(zero=False)
          ),
    y=
    alt.Y('y',
          scale=alt.Scale(zero=False)
          ),
    tooltip=['text']
).properties(
    width=700,
    height=400
)
chart.interactive()

# In[ ]:
