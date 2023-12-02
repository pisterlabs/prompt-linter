#!/usr/bin/env python
# coding: utf-8

# In[2]:


# get_ipython().system('pip install openai')
import os
import openai
import pandas as pd
import numpy as np
import time
import random
openai.organization = "INSERT ORG ID"
openai.api_key = "INSERT API KEY"


# In[3]:


ethos_data = pd.read_csv("inputs/Ethos_Dataset_Multi_Label.csv", delimiter=';')
ethos_data_binary = pd.read_csv("inputs/Ethos_Dataset_Binary.csv", delimiter=';')
sexist = pd.DataFrame(ethos_data[ethos_data["gender"] >= 0.5]["comment"])       # Filter ethos_data for gender score > 0.5, take only comment column (84 rows)
racist = pd.DataFrame(ethos_data[ethos_data["race"] >= 0.5]["comment"])         # Filter ethos_data for race score > 0.5, take only comment column (76 rows)
not_hate = ethos_data_binary[ethos_data_binary["isHate"] == 0]                  # Filter ethos_data_binary for non-hate comments (354 rows)
sexist["category"] = "sexist"       # Add "category" column to comments
racist["category"] = "racist"       # ""
not_hate["category"] = "not hate"   # ""
comments = pd.concat([sexist, racist])  # Create one dataframe called comments (160 rows)
comments = comments.reset_index()       # Resets row numbers to 0-159 (otherwise indices inherited from position in ethos_data)

### Other categories ###
#disability = ethos_data[ethos_data["disability"] >= 1]["comment"]
#religion = ethos_data[ethos_data["religion"] >= 1]["comment"]
#sexual_orientation = ethos_data[ethos_data["sexual_orientation"] >= 1]["comment"]


# In[ ]:


##### Zero-Shot #####
def zero_shot(cat, label, comment, temperature):
    prompt='Is the following text '+cat+', answer yes or no. '+' "'+comment+'" '+' Answer:'
    zero_shot_results = dict()
    response = openai.Completion.create(    # GPT-3 interaction
        engine="davinci",
        prompt=prompt,
        temperature=temperature,
        max_tokens=2,
        top_p=1,
        n = 1,
        stop=["\n"]
    )

    zero_shot_results["category"] = cat
    zero_shot_results["label"] = label
    zero_shot_results["prompt"] = prompt
    zero_shot_results['answer'] = response["choices"][0].text
    zero_shot_results['temperature'] = temperature
    return zero_shot_results    # Output is dict with keys as above


all_zero_shot_result = pd.DataFrame()   # Create empty data frame
# Sexist, Temperature = 0
j = 0
while j < 500:
    print("Zero shot learning, iteration {}".format(j))
    for i, sexist_comment in enumerate(sexist[0:30].comment):  # Loop through first 30 comments in sexist, where i is index number (0-29) and sexist_comment is comment as a string
        not_sexist_comment = not_hate.reset_index().comment[i]  # Extract comment i from not_hate
        zero_shot_sexist_result = zero_shot("sexist", "sexist", sexist_comment, temperature = 0.3)    # Run sexist_comment through GPT-3
        all_zero_shot_result = all_zero_shot_result.append(zero_shot_sexist_result, ignore_index=True)  # Add results for sexist_comment to all results
        zero_shot_sexist_result = zero_shot("sexist", "not-sexist", not_sexist_comment, temperature = 0.3)    # Run not_sexist_comment through GPT-3
        all_zero_shot_result = all_zero_shot_result.append(zero_shot_sexist_result, ignore_index=True)      # Add results for not_sexist_comment to all results

    # Racist, Temperature = 0
    for i, racist_comment in enumerate(racist[0:30].comment):
        not_racist_comment = not_hate.reset_index().comment[i]
        zero_shot_racist_result = zero_shot("racist", "racist", racist_comment, temperature = 0.3)
        all_zero_shot_result = all_zero_shot_result.append(zero_shot_racist_result, ignore_index=True)
        zero_shot_racist_result = zero_shot("racist", "not-racist", not_racist_comment, temperature = 0.3)
        all_zero_shot_result = all_zero_shot_result.append(zero_shot_racist_result, ignore_index=True)
    j += 1

# # Sexist, Temperature = 0.25
# for i, sexist_comment in enumerate(sexist[0:30].comment):
#     not_sexist_comment = not_hate.reset_index().comment[i]
#     zero_shot_sexist_result = zero_shot("sexist", "sexist", sexist_comment, temperature = 0.25)
#     all_zero_shot_result = all_zero_shot_result.append(zero_shot_sexist_result, ignore_index=True)
#     zero_shot_sexist_result = zero_shot("sexist", "not-sexist", not_sexist_comment, temperature = 0.25)
#     all_zero_shot_result = all_zero_shot_result.append(zero_shot_sexist_result, ignore_index=True)
#
# # Racist, Temperature = 0.25
# for i, racist_comment in enumerate(racist[0:30].comment):
#     not_racist_comment = not_hate.reset_index().comment[i]
#     zero_shot_racist_result = zero_shot("racist", "racist", racist_comment, temperature = 0.25)
#     all_zero_shot_result = all_zero_shot_result.append(zero_shot_racist_result, ignore_index=True)
#     zero_shot_racist_result = zero_shot("racist", "not-racist", not_sexist_comment, temperature = 0.25)
#     all_zero_shot_result = all_zero_shot_result.append(zero_shot_racist_result, ignore_index=True)


# In[5]:


all_zero_shot_result


# In[203]:


# One-Shot
def one_shot(cat, label, example, comment, temperature):
    one_shot_results = dict()
    prompt = 'The following text in quotes is ' + cat+ ': ' +        '"'+example + '"' +        " Is the following text in quotes " + cat+ ", answer yes or no: " +        '"'+comment + '."'+        " Answer:"
    response = openai.Completion.create(
        engine="davinci",
        prompt = prompt,
        temperature=temperature,
        max_tokens=2,
        top_p=1,
        n = 1,
        stop=["\n"]
    )
    one_shot_results["category"] = cat
    one_shot_results["label"] = label
    one_shot_results["prompt"] = prompt
    one_shot_results["example"] = example
    one_shot_results["comment"] = comment
    one_shot_results['answer'] = response["choices"][0].text
    one_shot_results['temperature'] = temperature

    return one_shot_results

all_one_shot_result = pd.DataFrame()
# Sexist, Temperature = 0
j = 0
while j < 100:
    print("One shot learning, iteration {}".format(j))
    for i, sexist_comment in enumerate(sexist[0:30].comment):
        not_sexist_comment = not_hate.reset_index().comment[i]
        example = sexist.reset_index().comment[i+30]
        one_shot_sexist_result = one_shot("sexist", "sexist", example, sexist_comment, temperature = 0.3)
        all_one_shot_result = all_one_shot_result.append(one_shot_sexist_result, ignore_index=True)
        one_shot_sexist_result = one_shot("sexist", "not-sexist", example, not_sexist_comment, temperature = 0.3)
        all_one_shot_result = all_one_shot_result.append(one_shot_sexist_result, ignore_index=True)

    # # Racist, Temperature = 0
    for i, racist_comment in enumerate(racist[0:30].comment):
        not_racist_comment = not_hate.reset_index().comment[i]
        example = racist.reset_index().comment[i+30]
        one_shot_racist_result = one_shot("racist", "racist", example, racist_comment, temperature = 0.3)
        all_one_shot_result = all_one_shot_result.append(one_shot_racist_result, ignore_index=True)
        one_shot_racist_result = one_shot("racist", "not-racist", example, not_racist_comment, temperature = 0.3)
        all_one_shot_result = all_one_shot_result.append(one_shot_racist_result, ignore_index=True)
    j += 1

# Sexist, Temperature = 0.25
# for i, sexist_comment in enumerate(sexist[0:30].comment):
#     not_sexist_comment = not_hate.reset_index().comment[i]
#     example = sexist.reset_index().comment[i+30]
#     one_shot_sexist_result = one_shot("sexist", "sexist", example, sexist_comment, temperature = 0.25)
#     all_one_shot_result = all_one_shot_result.append(one_shot_sexist_result, ignore_index=True)
#     one_shot_sexist_result = one_shot("sexist", "not-sexist", example, not_sexist_comment, temperature = 0.25)
#     all_one_shot_result = all_one_shot_result.append(one_shot_sexist_result, ignore_index=True)
#
# # Racist, Temperature = 0.25
# for i, racist_comment in enumerate(racist[0:30].comment):
#     not_racist_comment = not_hate.reset_index().comment[i]
#     example = racist.reset_index().comment[i+30]
#     one_shot_racist_result = one_shot("racist", "racist", example, racist_comment, temperature = 0.25)
#     all_one_shot_result = all_one_shot_result.append(one_shot_racist_result, ignore_index=True)
#     one_shot_racist_result = one_shot("racist", "not-sexist", example, not_sexist_comment, temperature = 0.25)
#     all_one_shot_result = all_one_shot_result.append(one_shot_racist_result, ignore_index=True)


# In[251]:


# Single-category few-shot
def few_shot_single(cat, label, example1, example2, example3, comment, temperature):
    few_shot_results = dict()
    prompt = "'''"+example1+"'''"+ ": " + cat+'. \n'+        "'''"+example2+"'''"+ ": " + "not-"+ cat +'. \n'+        "'''"+example3+"'''"+ ": " + cat+'. \n'+        "'''"+comment+"''':"
    response = openai.Completion.create(
        engine="davinci",
        prompt= prompt,
        temperature=temperature,
        max_tokens=4,
        top_p=1,
        n = 1,
        stop=["."]
    )
    few_shot_results["category"] = cat
    few_shot_results["label"] = label
    few_shot_results["prompt"] = prompt
    few_shot_results["example1"] = example1
    few_shot_results["example2"] = example2
    few_shot_results["example3"] = example3
    few_shot_results["comment"] = comment
    few_shot_results['answer'] = response["choices"][0].text
    few_shot_results["temperature"] = temperature

    return few_shot_results

all_few_shot_single_result = pd.DataFrame()
# Single Cat: Sexist, Temperature = 0
j = 0
while j < 100:
    print("Single category few-shot learning, iteration {}".format(j))
    for i, sexist_comment in enumerate(sexist[0:30].comment):
        not_sexist_comment = not_hate.reset_index().comment[i]  # Extract comment i from not_hate
        example1 = sexist.reset_index().comment[i+30]           # Extract comment i + 30 from sexist to use as example
        example2 = not_hate.reset_index().comment[i+30]         # Extract comment i + 30 from not_hate to use as example
        example3 = sexist.reset_index().comment[i+50]           # Extract comment i + 50 from sexist to use as example
        few_shot_sexist_result = few_shot_single("sexist", "sexist", example1, example2, example3, sexist_comment, temperature = 0.3)
        all_few_shot_single_result = all_few_shot_single_result.append(few_shot_sexist_result, ignore_index=True)
        few_shot_sexist_result = few_shot_single("sexist", "not-sexist", example1, example2, example3, not_sexist_comment, temperature = 0.3)
        all_few_shot_single_result = all_few_shot_single_result.append(few_shot_sexist_result, ignore_index=True)
    time.sleep(5)
    #Single Cat: Racist, Temperature = 0
    for i, racist_comment in enumerate(racist[0:30].comment):
        not_racist_comment = not_hate.reset_index().comment[i]
        example1 = racist.reset_index().comment[i+30]
        example2 = not_hate.reset_index().comment[i+30]
        example3 = racist.reset_index().comment[i+45]
        few_shot_racist_result = few_shot_single("racist", "racist", example1, example2, example3, racist_comment, temperature = 0.3)
        all_few_shot_single_result = all_few_shot_single_result.append(few_shot_racist_result, ignore_index=True)
        few_shot_racist_result = few_shot_single("racist", "not-racist", example1, example2, example3, not_racist_comment, temperature = 0.3)
        all_few_shot_single_result = all_few_shot_single_result.append(few_shot_racist_result, ignore_index=True)
    time.sleep(5)
    j += 1


# In[ ]:

# Mixed-category few-shot
def few_shot_mixed(cat, label, example1, example2, example3, comment, temperature):
    few_shot_results = dict()
    prompt = "'''"+example1+"'''"+ ": " + "sexist. \n"+        "'''"+example2+"'''"+ ": " + "racist. \n"+        "'''"+example3+"'''"+ ": " + "neither. \n"+        "'''"+comment+"''':"
    response = openai.Completion.create(
        engine="davinci",
        prompt= prompt,
        temperature=0,
        max_tokens=4,
        top_p=1,
        n = 1,
        stop=["."]
    )
    few_shot_results["category"] = cat
    few_shot_results["label"] = label
    few_shot_results["prompt"] = prompt
    few_shot_results["example1"] = example1
    few_shot_results["example2"] = example2
    few_shot_results["example3"] = example3
    few_shot_results["comment"] = comment
    few_shot_results['answer'] = response["choices"][0].text
    few_shot_results["temperature"] = temperature

    return few_shot_results

all_few_shot_mixed_result = pd.DataFrame()
#Mixed Cat: Sexist, Temperature = 0
for i, sexist_comment in enumerate(sexist[0:30].comment):
    neither_comment = not_hate.reset_index().comment[i+300]
    example1 = sexist.reset_index().comment[i+30]
    example2 = racist.reset_index().comment[i+30]
    example3 = not_hate.reset_index().comment[i+30]
    few_shot_mixed_result = few_shot_mixed("mixed", "sexist", example1, example2, example3, sexist_comment, temperature = 0)
    all_few_shot_mixed_result = all_few_shot_mixed_result.append(few_shot_mixed_result, ignore_index=True)
    few_shot_mixed_result = few_shot_mixed("mixed", "neither", example1, example2, example3, neither_comment, temperature = 0)
    all_few_shot_mixed_result = all_few_shot_mixed_result.append(few_shot_mixed_result, ignore_index=True)

#Mixed Cat: Racist, Temperature = 0
for i, racist_comment in enumerate(racist[0:30].comment):
    neither_comment = not_hate.reset_index().comment[i+100]
    example1 = sexist.reset_index().comment[i+30]
    example2 = racist.reset_index().comment[i+30]
    example3 = not_hate.reset_index().comment[i+30]
    few_shot_mixed_result = few_shot_mixed("mixed", "racist", example1, example2, example3, racist_comment, temperature = 0.5)
    all_few_shot_mixed_result = all_few_shot_mixed_result.append(few_shot_mixed_result, ignore_index=True)
    few_shot_mixed_result = few_shot_mixed("mixed", "neither", example1, example2, example3, neither_comment, temperature = 0.5)
    all_few_shot_mixed_result = all_few_shot_mixed_result.append(few_shot_mixed_result, ignore_index=True)


# In[ ]:

# Mixed-category few-shot, with instruction
def few_shot_instruction(cat, label, example1, example2, example3, comment, temperature):
    few_shot_results = dict()
    prompt = "Classify the following texts into 'racist', 'sexist' or 'neither'. \n" +        "'''"+example1+"'''"+ ": " + "sexist. \n"+        "'''"+example2+"'''"+ ": " + "racist. \n"+        "'''"+example3+"'''"+ ": " + "neither. \n"+        "'''"+comment+"''':"
    response = openai.Completion.create(
        engine="davinci",
        prompt= prompt,
        temperature=temperature,
        max_tokens=4,
        top_p=1,
        n = 1,
        stop=["."]
    )
    few_shot_results["category"] = cat
    few_shot_results["label"] = label
    few_shot_results["prompt"] = prompt
    few_shot_results["example1"] = example1
    few_shot_results["example2"] = example2
    few_shot_results["example3"] = example3
    few_shot_results["comment"] = comment
    few_shot_results['answer'] = response["choices"][0].text
    few_shot_results["temperature"] = temperature

    return few_shot_results

all_few_shot_instruction_result = pd.DataFrame()
#Mixed Cat: Sexist, Temperature = 0
for i, sexist_comment in enumerate(sexist[0:30].comment):
    neither_comment = not_hate.reset_index().comment[i+300]
    example1 = sexist.reset_index().comment[i+30]
    example2 = racist.reset_index().comment[i+30]
    example3 = not_hate.reset_index().comment[i+30]
    few_shot_mixed_result = few_shot_instruction("mixed", "sexist", example1, example2, example3, sexist_comment, temperature = 0)
    all_few_shot_instruction_result = all_few_shot_instruction_result.append(few_shot_mixed_result, ignore_index=True)
    few_shot_mixed_result = few_shot_instruction("mixed", "neither", example1, example2, example3, neither_comment, temperature = 0)
    all_few_shot_instruction_result = all_few_shot_instruction_result.append(few_shot_mixed_result, ignore_index=True)

#Mixed Cat: Racist, Temperature = 0
for i, racist_comment in enumerate(racist[0:30].comment):
    neither_comment = not_hate.reset_index().comment[i+100]
    example1 = sexist.reset_index().comment[i+30]
    example2 = racist.reset_index().comment[i+30]
    example3 = not_hate.reset_index().comment[i+30]
    few_shot_mixed_result = few_shot_instruction("mixed", "racist", example1, example2, example3, racist_comment, temperature = 0)
    all_few_shot_instruction_result = all_few_shot_instruction_result.append(few_shot_mixed_result, ignore_index=True)
    few_shot_mixed_result = few_shot_instruction("mixed", "neither", example1, example2, example3, neither_comment, temperature = 0)
    all_few_shot_instruction_result = all_few_shot_instruction_result.append(few_shot_mixed_result, ignore_index=True)
all_few_shot_instruction_result


# In[21]:

# Mixed-category few-shot
import time
def few_shot_fixed_examples(i, cat, label, comment, temperature):
    few_shot_results = dict()
    example1 = sexist.reset_index()["comment"][i]
    example2 = racist.reset_index()["comment"][i]
    example3 = not_hate.reset_index()["comment"][i]
    prompt = "'''"+example1+"'''"+ ": " + "sexist. \n"+        "'''"+example2+"'''"+ ": " + "racist. \n"+        "'''"+example3+"'''"+ ": " + "neither. \n"+        "'''"+comment+"''':"
    response = openai.Completion.create(
        engine="davinci",
        prompt= prompt,
        temperature=temperature,
        max_tokens=4,
        top_p=1,
        n = 1,
        stop=["."]
    )
    few_shot_results["category"] = cat
    few_shot_results["label"] = label
    few_shot_results["prompt"] = prompt
    few_shot_results["example1"] = example1
    few_shot_results["example2"] = example2
    few_shot_results["example3"] = example3
    few_shot_results["example_set"] = i
    few_shot_results["comment"] = comment
    few_shot_results['answer'] = response["choices"][0].text
    few_shot_results["temperature"] = temperature

    return few_shot_results

all_few_shot_fixed_examples_result = pd.read_csv("outputs/data/few_shot_fixed_examples_results.csv")

#Mixed Cat, Temperature = 0
for i in range(0, 10):
    print("Few shot multi category, iteration {}".format(i))
    neither_comments = not_hate.reset_index().comment[0:121]
    sexist_comments = sexist.reset_index().comment[0:61]
    racist_comments = racist.reset_index().comment[0:61]
    for racist_comment in racist_comments.drop(i):
        few_shot_fixed_examples_result = few_shot_fixed_examples(i, "fixed-example", "racist", racist_comment, temperature = 0.3)
        all_few_shot_fixed_examples_result = all_few_shot_fixed_examples_result.append(few_shot_fixed_examples_result, ignore_index=True)
        time.sleep(2)
    for sexist_comment in sexist_comments.drop(i):
        few_shot_fixed_examples_result = few_shot_fixed_examples(i, "fixed-example", "sexist", sexist_comment, temperature = 0.3)
        all_few_shot_fixed_examples_result = all_few_shot_fixed_examples_result.append(few_shot_fixed_examples_result, ignore_index=True)
        time.sleep(2)
    for neither_comment in neither_comments.drop(i):
        few_shot_fixed_examples_result = few_shot_fixed_examples(i, "fixed-example", "neither", neither_comment, temperature = 0.3)
        all_few_shot_fixed_examples_result = all_few_shot_fixed_examples_result.append(few_shot_fixed_examples_result, ignore_index=True)
        time.sleep(2)


def few_shot_instruction_fixed_examples(i, cat, label, comment, temperature):
    few_shot_results = dict()
    example1 = sexist.reset_index()["comment"][i]
    example2 = racist.reset_index()["comment"][i]
    example3 = not_hate.reset_index()["comment"][i]
    prompt = "Classify the following texts into 'racist', 'sexist' or 'neither'. \n" +        "'''"+example1+"'''"+ ": " + "sexist. \n"+        "'''"+example2+"'''"+ ": " + "racist. \n"+        "'''"+example3+"'''"+ ": " + "neither. \n"+        "'''"+comment+"''':"
    response = openai.Completion.create(
        engine="davinci",
        prompt= prompt,
        temperature=temperature,
        max_tokens=4,
        top_p=1,
        n = 1,
        stop=["."]
    )
    few_shot_results["category"] = cat
    few_shot_results["label"] = label
    few_shot_results["prompt"] = prompt
    few_shot_results["example1"] = example1
    few_shot_results["example2"] = example2
    few_shot_results["example3"] = example3
    few_shot_results["example_set"] = i
    few_shot_results["comment"] = comment
    few_shot_results['answer'] = response["choices"][0].text
    few_shot_results["temperature"] = temperature

    return few_shot_results

all_few_shot_instruction_fixed_examples_result = pd.read_csv("outputs/data/few_shot_fixed_examples_instruction_results.csv")

#Mixed Cat, Temperature = 0
for i in range(0, 10):
    neither_comments = not_hate.reset_index().comment[0:121]
    sexist_comments = sexist.reset_index().comment[0:61]
    racist_comments = racist.reset_index().comment[0:61]
    print("Few shot mixed category w/ instruction, iteration {}".format(i))
    for racist_comment in racist_comments.drop(i):
        few_shot_instruct_fixed_examples_result = few_shot_instruction_fixed_examples(i, "fixed-example", "racist", racist_comment, temperature = 0.3)
        all_few_shot_instruction_fixed_examples_result = all_few_shot_instruction_fixed_examples_result.append(few_shot_instruct_fixed_examples_result, ignore_index=True)
        time.sleep(2)
    for sexist_comment in sexist_comments.drop(i):
        few_shot_instruct_fixed_examples_result = few_shot_instruction_fixed_examples(i, "fixed-example", "sexist", sexist_comment, temperature = 0.3)
        all_few_shot_instruction_fixed_examples_result = all_few_shot_instruction_fixed_examples_result.append(few_shot_instruct_fixed_examples_result, ignore_index=True)
        time.sleep(2)
    for neither_comment in neither_comments.drop(i):
        few_shot_instruct_fixed_examples_result = few_shot_instruction_fixed_examples(i, "fixed-example", "neither", neither_comment, temperature = 0.3)
        all_few_shot_instruction_fixed_examples_result = all_few_shot_instruction_fixed_examples_result.append(few_shot_instruct_fixed_examples_result, ignore_index=True)
        time.sleep(2)


pre_all_few_shot_fixed_examples_result = pd.read_csv("../outputs/data/few_shot_fixed_examples_results.csv")
pre_all_few_shot_fixed_examples_result = pre_all_few_shot_fixed_examples_result.append(all_few_shot_fixed_examples_result)
# pre_all_few_shot_fixed_examples_result


# In[27]:


# all_zero_shot_result.to_csv("outputs/data/zero_shot_results.csv")
all_one_shot_result.to_csv("outputs/data/one_shot_results_multiple.csv")
all_few_shot_single_result.to_csv("outputs/data/few_shot_single_results_multiple.csv")
# all_few_shot_mixed_result.to_csv("../outputs/data/few_shot_mixed_results.csv")
all_few_shot_instruction_fixed_examples_result.to_csv("outputs/data/few_shot_fixed_example_instruction_results.csv")
all_few_shot_fixed_examples_result.to_csv("outputs/data/few_shot_fixed_examples_results_multiple.csv")


# In[ ]:




