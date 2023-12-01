import json
import os 
from bs4 import BeautifulSoup
import openai
import random
import re
import shutil
dataPath = "chainofaction/data/parsed_solutions"
files = os.listdir(dataPath)
""" 
for file in files:
    with open(os.path.join(dataPath,file)) as f:
        try:
            data = json.load(f)
            key = list(data["solutions"].keys())[0]
            text_content = data["solutions"][key]["codeblocks_text"][0]

            # Adding newline characters after each paragraph
            #  soup = BeautifulSoup(text_content, 'html.parser')
            #text_content = soup.get_text(separator='', strip=False)    
            #
            if text_content.startswith("class Solution:"):
                with open("chainofaction/data/code/"+file+".py","w") as f:
                    f.write(text_content)
        except Exception as e:
            print(data.keys(),f)

 

for file in files:
    with open(os.path.join(dataPath,file)) as f:
        try:
            data = json.load(f)
            text = data["problem_text"]

            # Adding newline characters after each paragraph
            #  soup = BeautifulSoup(text_content, 'html.parser')
            #text_content = soup.get_text(separator='', strip=False)    
            #
            if file+".py" in os.listdir("chainofaction/data/code"):
                with open("chainofaction/data/problems/"+file+".txt","w") as f:
                    f.write(text)
        except Exception as e:
            print(data.keys(),f)



def extract_testcases(text):
    pattern = re.compile(r'Input: ([^\n]+)\s+Output: ([^\n]+)', re.MULTILINE | re.DOTALL)
    matches = pattern.findall(text)
    testcases = [(input_str.strip(), output_str.strip()) for input_str, output_str in matches]
    return testcases



#This generates testcase inputs, after given the problem and the solution
def generate_testcases(text,soln):
    prompt = f"This is the problem: {text}\n\nThis is the solution: {soln}\n\nPlease generate 5 inputs for the solution of this question. Format will be Input: <testcase>. You are to only print the inputs, no other words. Wrap testcases with multiple inputs in (), separated by commas"
    text = openai.ChatCompletion.create(
        model = "gpt-4-1106-preview",
        max_tokens = 300,
        messages = [{"role":"user", "content": prompt}],
        temperature = 0.5
    )['choices'][0]["message"]["content"]

    return text

for file in files:
  with open(os.path.join(dataPath,file)) as f:
      data = json.load(f)
      # print(data.keys())
      if "problem_text" not in data or "solutions" not in data:
        continue
      text = data['problem_text']
      solns = data['solutions']
    # num_soln = data['n_solutions']
      # print(text)
      # print(testcases)
     
      # else:
      for sol in solns:
        #retrieve plaintext file
        with open(f"chainofaction/data/code/{file}.py",'w') as f:
          soln = solns[sol]['codeblocks_text-only'][0]
          if soln.startswith("class Solution:"):
            f.write(soln)
            with open(f"chainofaction/data/problems/{file}.txt","w") as f:
                f.write(text)
            #get test cases if no initial testcases
            with open(f"chainofaction/data/cases/{file}.txt","w") as f:
                f.write(generate_testcases(text,soln))
              
                
            #get function head here
            pattern = r"def [^\n]*\n"
            matches = re.findall(pattern, soln)
            if matches:
                for match in matches:
                    print(match.strip())
            break
          else:
            continue
 """
""" for i in os.listdir("chainofaction/data/code")[173:]:
    with open("chainofaction/data/code/"+i) as f:
        code = f.read()
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[{"role":"user","content":"Describe the code and what it does:\n\n"+code+"\n\n in 100-200 tokens"}],
            max_tokens=300,
            temperature=0
        )["choices"][0]["message"]["content"]


    with open("chainofaction/data/descriptions/"+i[:-2]+"txt","w") as f:
        f.write(response)

 """

import pandas as pd
texts = []
titles = []
problems = []
#Takes problem, solution and description and turn into csv file
allcases = os.listdir("chainofaction/data/fullcases")
for i in os.listdir("chainofaction/data/descriptions"):
  if i[:-3]+"json" not in allcases:
     continue
  with open("chainofaction/data/descriptions/"+i) as f:
    text = f.read()
    texts.append(text)
  titles.append( i)
  with open("chainofaction/data/problems/"+i[:-3]+"txt") as f:
    problem = f.read()
    problems.append(problem)

  df = pd.DataFrame({"problem_text":problems,"title":titles,"skill_description":texts})
  #Convert into tsv file called leetcode.tsv. I need an id column by renaming index to id
df.to_csv("chainofaction/data/leetcode.tsv",sep="\t",index_label="id") 