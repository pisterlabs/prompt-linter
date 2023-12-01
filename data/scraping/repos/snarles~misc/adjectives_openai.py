import os
import openai
import numpy as np
from numpy.random import choice
from scipy.stats import rankdata

openai.api_key = ''

# Filter for personality adjs

adjs = np.loadtxt('talkenglish_adj_list.txt', dtype=str, delimiter = ':')

counts = np.zeros(len(adjs))
hits = np.zeros(len(adjs))

for i in range(100):
    rand_inds = choice(len(adjs), 50, replace = False)
    counts[rand_inds] += 1

    content = "Which of the following adjectives are commonly used to describe personality traits or emotional states? " + ', '.join(adjs[rand_inds])

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "user",
          "content": content
        }
      ],
      temperature=1,
      max_tokens=500,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    desc = response["choices"][0]["message"]["content"]

    wds = desc.replace(' ','').split(',')
    for wd in wds:
        if wd in adjs[rand_inds]:
            hits[adjs == wd] += 1

scores = (hits + 1)/(counts + 2)

pads = adjs[np.argsort(-scores)][0:100]
np.savetxt("selected_adjs.txt", pads, fmt = "%s")

# find adjs which are associated with ??
# assoc with hide

counts = np.zeros(len(pads))
hits = np.zeros(len(pads))


rand_inds = choice(len(pads), 20, replace = False)
counts[rand_inds] += 1
content = "Which of the following words suggest deceptivenes/shyness/reclusiveness/passive? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] += 1

content = "Which of the following words suggest boldness/openness/adventurousness/braveness/proactive? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] -= 1



h_scores = (hits)/(counts + 1)
pads[np.argsort(-h_scores)]

# F scores
counts = np.zeros(len(pads))
hits = np.zeros(len(pads))


rand_inds = choice(len(pads), 20, replace = False)
counts[rand_inds] += 1
content = "Which of the following words suggest empathy/ability to understand others? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] += 1

content = "Which of the following words suggest aloofness/indifference/lack of empathy? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] -= 1



f_scores = (hits)/(counts + 1)
pads[np.argsort(-f_scores)]


# L scores
counts = np.zeros(len(pads))
hits = np.zeros(len(pads))


rand_inds = choice(len(pads), 20, replace = False)
counts[rand_inds] += 1
content = "Which of the following words suggest high status/attractive/impressive/flashy? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] += 1

content = "Which of the following words suggest low status/unattractive/ordinary/repulsive? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] -= 1



l_scores = (hits)/(counts + 1)
pads[np.argsort(-l_scores)]

# P scores
counts = np.zeros(len(pads))
hits = np.zeros(len(pads))


rand_inds = choice(len(pads), 20, replace = False)
counts[rand_inds] += 1
content = "Which of the following words suggest desire/hunger/neediness/passion/compassion/obsession/feeling/emotionality? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] += 1

content = "Which of the following words suggest apathy/unmotivated? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] -= 1



p_scores = (hits)/(counts + 1)
pads[np.argsort(-p_scores)]

# R scores
counts = np.zeros(len(pads))
hits = np.zeros(len(pads))


rand_inds = choice(len(pads), 20, replace = False)
counts[rand_inds] += 1
content = "Which of the following words suggest knowledge/wisdom/memory/planning? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] += 1

content = "Which of the following words suggest ignorance/instinct/confusion? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] -= 1



r_scores = (hits)/(counts + 1)
pads[np.argsort(-r_scores)]

# S scores

counts = np.zeros(len(pads))
hits = np.zeros(len(pads))


rand_inds = choice(len(pads), 20, replace = False)
counts[rand_inds] += 1
content = "Which of the following words suggest efficiency/speed/reactivity/precision? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] += 1

content = "Which of the following words suggest deliberation/inefficiency/slowness/slackness? " + ', '.join(pads[rand_inds])
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": content
    }
  ],
  temperature=1,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
desc = response["choices"][0]["message"]["content"]
desc = desc.replace("\n- ", ", ")
wds = desc.replace(' ','').split(',')
for wd in wds:
    if wd.lower() in pads[rand_inds]:
        hits[pads == wd.lower()] -= 1



s_scores = (hits)/(counts + 1)
pads[np.argsort(-s_scores)]


# putting them together
arr = np.array([
    h_scores,
    f_scores,
    l_scores,
    p_scores,
    r_scores,
    s_scores,
]).T
for i in range(6):
    arr[:, i] = rankdata(arr[:, i])

for i in range(len(pads)):
    st = pads[i] + ","
    v = arr[i]
    v = v/np.sum(v) * 20
    v = np.floor(v).astype(int)
    for j in range(6):
        st = st + str(v[j]) + ','
    print(st[:-1])
