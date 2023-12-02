#!/usr/bin/env python
# coding: utf-8

# In[28]:


import openai
from pprint import pprint
import time
import random
import datetime

date_time = datetime.datetime.now()
d = date_time.strftime("%s")


# In[2]:


openai.api_key = ""

def call_model(messages):
    completion = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      temperature = 0.8,
      messages = messages
    )
    
    return completion.choices[0].message.content

economic_industries = [
    "Agriculture",
    "Automotive",
    "Banking",
    "Construction",
    "Education",
    "Energy (Oil & Gas)",
    "Entertainment",
    "Fashion",
    "Finance",
    "Food and Beverage",
    "Healthcare",
    "Hospitality",
    "Information Technology",
    "Insurance",
    "Manufacturing",
    "Media",
    "Mining",
    "Pharmaceuticals",
    "Real Estate",
    "Retail",
    "Telecommunications",
    "Transportation",
    "Travel and Tourism",
    "Utilities (Water, Electricity, Gas)",
    "Waste Management",
    "Aerospace",
    "Biotechnology",
    "Chemical",
    "Forestry",
    "Maritime",
    "Nuclear Energy",
    "Renewable Energy",
    "Space Exploration",
    "Sports",
    "Textiles",
    "Forex and Trading",
    "E-commerce",
    "Digital Marketing",
    "Art and Culture",
    "Advertising",
    "Telecommunication Equipment",
    "Computer Hardware",
    "Software Development",
    "Pharmaceutical Research",
    "Agricultural Machinery",
    "Fishery",
    "Automotive Parts",
    "Film and Television Production",
    "Online Gaming",
    "Airlines",
    "Cruise Lines",
    "Rail Transportation",
    "Shipping",
    "Wind Energy",
    "Solar Energy",
    "Nuclear Technology",
    "Music",
    "Publishing",
    "Chemical Engineering",
]


# In[6]:


def create_random_prompt(chapter, roles=["Customer", "Salesman"], range_vals=(3, 7), industries=None):
    if industries is None:
        industries = ["tech", "health", "finance"]  # default industries; replace with your default list if different
    
    x = random.randint(*range_vals)
    
    y = 0
    for i in reversed(range(3, 9)):  # Generalized loop for range of values
        if i * x < 27:
            y = i
            break

    conversation_structure = ""
    for i in range(1, x+1):
            conversation_structure += f"""
        {roles[0]}: #{i}. sentence of {roles[0].lower()}
        {roles[1]}: #{i}. sentence of {roles[1].lower()}"""

    prompt = f"""Here is a chapter from a textbook about convincing people. 
    The purpose of this data is to use it to fine tune a llm. 
    Generate conversation examples that are based on the chapter that is provided and would help an ai to learn the topic by examples. 
    Focus only on the topic that is given in the chapter when generating the examples. 
    Let the example be in the {random.choice(industries)} industry.

    Follow this structure and put each conversation in a list of objects in json format. Only return the json nothing more:
    {conversation_structure}

    Generate {y} lists of those conversations

    Chapter:{chapter}"""

    return prompt


# In[17]:


def get_number_of_points(answer, key="answer"):
    """Fetch the number of points from the given key in the JSON answer."""
    answer = answer.replace("'", "").replace("\n", "")
    answer_data = json.loads(answer)
    
    return len(answer_data.get(key, []))


def make_chat(prompts, answer_key="answer", detailed_prompt_template=None):
    """Interact with the model based on provided prompts and return an extended structure."""
    mgs = []
    ext_structure = []
    
    # Check if a custom template for the detailed prompt is provided
    if not detailed_prompt_template:
        detailed_prompt_template = (f"""Write a more detailed outline only for point {{index}}. And only """
                                    """return the json structure for point {index} don't return the other points. """
                                    """Write very detailed. The aim to make the reader a perfect salesman that can """
                                    """sell everything. Extend the points A, B, C etc and add new ones on this level. """
                                    """Stay in the json format. Here is the json outline: {outline}""")
    
    for prompt in prompts:
        m = {"role": "user", "content": prompt}
        mgs.append(m)
        a = call_model(mgs)
        mgs.append({"role": "assistant", "content": a})
        
        if prompt == prompts[-1]:
            n = get_number_of_points(a, key=answer_key)
            a = a.replace("'", "").replace("\n", "")
            answer_data = json.loads(a)
            
            for i in range(n):
                subpoint = answer_data[answer_key][i]
                
                detailed_prompt = detailed_prompt_template.format(index=i+1, outline=json.dumps(answer_data))
                ext_subpoint = call_model([{"role": "user", "content": detailed_prompt}])
                ext_structure.append(ext_subpoint)
                time.sleep(2)
    
    return ext_structure


def generate_textbook():
    headlines = [
        "Building Rapport and Capturing Attention",
        "Developing Exceptional Communication Skills",
        "Discovering Customer Needs and Pain Points",
        "Presenting Solutions and Benefits",
        "Overcoming Resistance and Objections",
        "Closing the Sale"
    ]

    structure = []

    # Generate subpoints for each headline
    for headline in headlines:
        prompt = f"""I want an outline for a textbook on the topic of sales and convincing people. Here is a list of the headlines:
        {headlines}
        
        Give me a very detailed and specific outline for the following headline: {headline}.
        Ignore everything related to body language and non-verbal communication, leave out everything that a computer could
        not perform like body language or non-verval communication.
        Return the answer in the following json structure:
        "headline": "{headline}", subpoints: [list all the subpoints in this list]
        """
        
        inp = {
            "role": "user", 
            "content": prompt
        }
        
        x = call_model([inp])
        x = x.replace("'", "").replace("\n", "")
        x = json.loads(x)
        structure.append(x)

    # Generate detailed text for each subpoint
    textbook = []
    for headline in structure:
        subheadlines = headline["subpoints"]
        for subheadline in subheadlines:
            print(f"generting subheadline: {subheadline}")
            prompt = f"""
            I want to write a book about sales and convincing techniques. Here is the outline of the chapters:
            1. Building Rapport and Capturing Attention
            2. Developing Exceptional Communication Skills
            3. Discovering Customer Needs and Pain Points
            4. Presenting Solutions and Benefits
            5. Overcoming Resistance and Objections
            6. Closing the Sale
            
            Here is the outline of the current chapter that:        
            {headline}
            
            Write me a long and detailed text for the subpoint: {subheadline} of the current chapter and only write a text for this subpoint. 
            Ignore points like body language or tone of voice. Focus on the 
            Start by mentioning the Chapter and the subpoint.
            The overall aim is to write a textbook.
            to teach someone with less experience how to convince people and sell stuff.
            """
            
            text = call_model([{"role": "user", "content": prompt}])
            textbook.append(text)
            
            
    texts = ""
    for text in textbook:
        texts += f"{text} "
    
    with open(f"textbook_{d}.txt", "w") as textfile:
        textfile.write(texts)
    
    return textbook


# In[18]:


import json

def generate_examples(textbook, num_samples=100, output_filename='data.json'):
    """
    Generate examples based on the provided textbook data and save to a JSON file.
    
    Parameters:
    - textbook: List of texts for which examples are generated.
    - num_samples: Number of samples to generate for each text.
    - output_filename: Name of the output file where examples will be saved.
    """
    
    examples = {"data": []}

    for text in textbook:
        print(f"generating samples for: {text[:30]}")
        for i in range(0, num_samples):
            try:
                prompt = create_random_prompt(text)  # Assuming create_random_prompt is a function in your environment

                text_response = call_model([{"role": "user", "content": prompt}])
                x = text_response.replace("'", "").replace("\n", "")
                x = json.loads(x)
                examples["data"].append(x)
            except Exception as e:
                print("invalid json was return from openai")
            

    with open(output_filename, 'w') as f:
        json.dump(examples, f)

# Example usage:
#create textbook and save it as txt
textbook_data = generate_textbook()

#generate samples from textbook
generate_examples(textbook_data, num_samples=100, output_filename=f'data/output_data_{d}.json')


