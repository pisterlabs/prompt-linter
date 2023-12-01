from joblib import Parallel, delayed
import openai
openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:8000/v1"
import os
model = "vicuna-33b-v1.3"

def generate_q(details):
    try: 
        
        wiki_text = details['text']
        category = details['section']
        pageid = details['pageid']
        temperature = details['temperature']
        openai.api_key = "EMPTY" # Not support yet
        openai.api_base = "http://localhost:8000/v1"
        folder = f"./wiki/qa/{category}"
        os.makedirs(f"{folder}", exist_ok=True)
        if os.path.exists(f"{folder}/{pageid}.txt") == True:
            print("Exists", f"{folder}/{pageid}.txt")
            return None
        model = "vicuna-33b-v1.3"

        completion = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": """Step 1: From the given Wikipedia scientific text and construct a question related to the text that contains between 5 to 25 words.
        Step 2: Based on the question generated in Step 1 and the provided scientific text, formulate a correct answer that ranges between 10 to 30 words. Then, devise a similar alternative answer with the same word limit.
        Step 3: Generate additional wrong answers, each within the 10 to 30 words limit, that bear a close resemblance to the correct answer formulated in Step 2.
        Step 4: Ensure that the wrong answers from Step 3 exhibit various degrees of accuracy and bear a high similarity (on a scale from 1 to 100) to the question.
        Step 5: Verify that the correct answer and all the wrong answers contain a similar number of words.
        Step 6: Evaluate the similarity between the question and the correct answer, and provide a similarity score on a scale from 1 to 100.
        Step 7: Evaluate the similarity between the question and each wrong answer, and assign a similarity score on a scale from 1 to 100.
        Step 8: Format the final output to include the question, correct answer(s) with score(s), and wrong answers with their corresponding similarity scores. The format should be as follows:

        Question: ...
        Correct Answer 1: ...
        Correct Answer 1 Score: ...
        Correct Answer 2: ...
        Correct Answer 2 Score: ...
        Wrong Answer 1: ...
        Wrong Answer 1 Score: ...
        Wrong Answer 2: ...
        Wrong Answer 2 Score: ...
        Wrong Answer 3: ...
        Wrong Answer 3 Score: ...
        Wrong Answer 4: ...
        Wrong Answer 4 Score: ...
        Wrong Answer 5: ...
        Wrong Answer 5 Score: ...
        Wrong Answer 6: ...
        Wrong Answer 6 Score: ...
        Wrong Answer 7: ...
        Wrong Answer 7 Score: ...
        Wrong Answer 8: ...
        Wrong Answer 8 Score: ...
        Wrong Answer 9: ...
        Wrong Answer 9 Score: ...

        Sample Wikipedia Scientific Text:
        """+wiki_text.strip()},
        {"role":"user", "content": "Step 8 Output:"}            
        ]
        )
        output = completion.choices[0].message.content
        if 'Correct Answer 2'.lower() in output.lower():
            print(f"{folder}/{pageid}.txt")
            with open(f"{folder}/{pageid}.txt", "w") as f:
                f.write(output)
        else:
            print("Not Generated", f"{folder}/{pageid}.txt")
    except Exception as e:
        print(e)
        pass
    return None

#numbers = range(4)
#squared_numbers = Parallel(n_jobs=-4)(delayed(square_number)(i) for i in numbers)
#print(squared_numbers)
from glob import glob
from tqdm import tqdm
import json
all_sections = glob("/datadrive1/wiki/articles/*")

todo = []
for section in all_sections:
    count = 0
    all_articles = glob(f"{section}/*.json")
    print("\n Total",section, len(all_articles))
    for article_path in tqdm(all_articles, total=len(all_articles)):
        #print(article_path)
        with open(article_path,'r') as f:
            content = f.read()
            try: 
                content = json.loads(content)
            except:
                continue
            pageid = content['pageid']
            section = section
            paragraphs = content['text'].split("\n\n")
            blocks = []
            maximum_words = 512
            current_text = ""
            for paragraph in paragraphs:
                if len(current_text.split(" ")) < maximum_words:
                    current_text = current_text + "\n\n" + paragraph
                else:
                    break
            current_text = " ".join(current_text.split()[0:maximum_words])
            #print(" Total ", len(current_text.split(" ")))
            try:
                json.loads(content)
                count += 1
            except:
                pass
        todo.append({
            "pageid": pageid,
            "section": section,
            "text": current_text,
            "temperature": 0,
            "max_tokens": 1024
        })
    print("\n Fine", section, "--->", count)
        #data = json.load(open(article_path,'r'))
        #print(article_path, data)
    break
    #print(section)
len(todo)

#generate_q(todo[0])
Parallel(n_jobs=3)(delayed(generate_q)(details) for details in tqdm(todo,total=len(todo)))
