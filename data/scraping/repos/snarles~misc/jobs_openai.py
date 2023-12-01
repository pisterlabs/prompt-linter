import os
import openai
import numpy as np
from numpy.random import choice

openai.api_key = str(np.loadtxt("/Users/zhengchy/private/oai.txt", dtype=str))

jobs = np.loadtxt('wikilist_jobs.txt', dtype=str, delimiter = ':')

score_fns = np.array([ff for ff in os.listdir("jobs_openai/") if ff[:5]=='score'])
score_is = np.array([f.split('_')[0][6:] for f in score_fns]).astype(int)
job_is = np.unique(score_is)

rand_inds = np.sort(choice(len(jobs), 5, replace = False))
rand_inds = np.array([v for v in rand_inds if not v in job_is])
len(rand_inds)

for i in rand_inds:
    for j in range(3):

        job = jobs[i]

        content = "Imagine a character who is a %s.  Come up with a name and gender, and list 10 character traits or quirks, both positive and negative.  Then come up with a paragraph that illustrates a typical event in the life of that character." % job

        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {
              "role": "user",
              "content": content
            }
          ],
          temperature=1,
          max_tokens=1000,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )

        desc = response["choices"][0]["message"]["content"]
        para = desc.split('\n')[-1]

        content = "I’m writing a story about a person.  Here is a portion from my story.\n\n%s\n\nNow please summarize the personality described into a 6-number summary.  Each of the 6 numbers is from 1 to 100.  The first number is the character's deceptiveness or shyness (it will be higher if they are both deceptive and shy.  If only deceptive and not shy, it will be moderately high.  If only shy and not deceptive, moderately high.).  The second number is their ability to empathize and understand other people.  The third number is their ability to impress strangers.  The fourth number is the intensity of their passion or desire.  The fifth number is their extent of knowledge.  The sixth number is their efficiency at doing tasks, their reactiveness to surprises and their intelligence at learning new things.   Please give your ratings and explain your reasoning for each rating." % para

        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {
              "role": "user",
              "content": content
            }
          ],
          temperature=1,
          max_tokens=1000,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )

        desc2 = response["choices"][0]["message"]["content"]
        j1 = j
        while "scores%i_rep%i.txt" % (i, j1) in os.listdir("jobs_openai/"):
            j1 += 1
        np.savetxt("jobs_openai/story%i_rep%i.txt" % (i, j1), [desc], fmt = '%s')
        np.savetxt("jobs_openai/scores%i_rep%i.txt" % (i, j1),  [desc2], fmt = '%s')
