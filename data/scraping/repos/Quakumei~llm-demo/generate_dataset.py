#!/usr/bin/env python
# coding: utf-8

# In[45]:


from pdfminer.high_level import extract_text
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# In[46]:


PDF_FILE = "./ukodeksrf.pdf"


# In[47]:


get_ipython().run_cell_magic('time', '', 'text = extract_text(PDF_FILE)\n')


# In[48]:


get_ipython().run_cell_magic('time', '', 'footer = """Бесплатная юридическая консультация по телефонам: \n• 8 (499) 938-53-71 (Москва и МО); \n• 8 (812) 467-95-28 (Санкт-Петербург и ЛО); \n• 8 (800) 301-92-12 (Регионы РФ). \n\nКомментарии к статьям на сайте \nhttps://ukodeksrf.ru/"""\nprint(f"Before: {len(text)}")\ntext = text.replace(footer, \'\')\nprint(f"After: {len(text)}")\n')


# In[49]:


articles = [f"Статья {art}" for art in text.split("Статья")[1:]]


# In[55]:


df = pd.DataFrame(articles, columns=["article"])
print("Length of texts characteristics")
df['article'].map(lambda x: len(x)).describe()


# In[56]:

def article_to_prompt(article:str):
    return f"<|im_start|> system Ссылайся на источник при ответе. Например: <Твой ответ> (Статья N УК РФ). Используй только русский язык. НЕ используй английский или другие языки. <|im_end|><|im_start|> user Дан отрывок текста: '{article}'.\nПридумай два вопроса, на которые можно ответить с помощью этого текста. Например:\n1. Какой срок заключения за убийство?\nОтвет: За убийство одного человека без дополнительных квалифицирующих признаков грозит тюремный срок от 6 до 15 лет с возможным ограничением свободы до двух лет. (Статья 105 УК РФ)<|im_end|><|im_start|> assistant "

def article_to_messages(article:str):
    return [
            {"role":"system", "content":"Ты приводишь только ответ, кратко но полно" },
            {"role":"user", "content":f"Дан отрывок текста: '{article}'. Придумай два вопроса, на которые можно ответить с помощью этого текста."}
    ]
# In[57]:


df['text'] = df['article'].map(article_to_prompt)
print(df.head())


# In[54]:


import openai
# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

def get_generation(prompt:str):
    completion = openai.Completion.create(model="Open-Orca/Mistral-7B-OpenOrca",
            prompt=prompt, max_tokens=2048)
    #print("Completion result:", completion)
    #print(f"Prompt: {prompt}")
    try:
        print(completion['choices'][0]['text'])
        return completion['choices'][0]['text']
    except:
        return ''
# df = df.head(5)
df['qa'] = df['text'].progress_apply(get_generation)
df.to_csv('dataset.csv')

# In[ ]:
