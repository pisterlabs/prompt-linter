# make sure to:
# pip install cohere altair
# pip install scikit-learn

import requests
import cohere
import random
import pandas as pd
import numpy as np
import sklearn
# import seaborn as sns
# import altair as alt
co = cohere.Client('L2VMOXwleskZQjVuP5QEe2puJTKNLAzGaRhSEVTK')
api_key = 'L2VMOXwleskZQjVuP5QEe2puJTKNLAzGaRhSEVTK'

# Questions and answers are here for testing purposes.
# In the question-gen.py file, the q_s and a_s arrays are generated from the user input and should be used in place here.
# These are just arrays from one iteration of running question-gen.py
q_s = ['Question: What is the main idea of this lesson?', 'Question: Why is it expensive to connect and disconnect from a database multiple times?', 'Question: What is the Singleton Pattern?', 'Question: Why is it beneficial for the environment to save energy and processing power?']
a_s = ['Answer: Connecting to databases can be expensive in terms of time and resources, therefore, it is efficient to connect once and perform all the necessary functions dealing with the database.', 'Answer: Because it requires spending time and company resources for each connection and disconnection.', 'Answer: The Singleton Pattern is a design pattern that restricts the instantiation of a class to one object. It is often used for database connections.', 'Answer: Because the excessive connections and disconnections to the database are avoided. This, in turn, reduces the carbon footprint.']

def generate_text(prompt, temp=0):
  response = co.generate(
    model='command',
    prompt=prompt,
    max_tokens=200,
    temperature=temp)
  return response.generations[0].text


# Cohere API endpoint and API key
cohere_api_endpoint = "https://api.cohere.ai/v1/embed"
api_key = "L2VMOXwleskZQjVuP5QEe2puJTKNLAzGaRhSEVTK"

# output question
counter = len(q_s)
encourage = "Write a short encouraging phrase 2-10 words long to encourage a student to keep going! Be creative."
sadprompt = "Write a short encouraging phrase 2-10 words long telling a student that they got the answer wrong, but motivate them to keep going. Be creative."
hype_file_path = "src/qs/hype.txt"
sad_file_path = "src/qs/sad.txt"

for i in range(counter):
  if q_s[i] and a_s[i]:
    print(q_s[i])
    usr_ans = input("Answer TutorBo! ")
    real_ans = (a_s[i])

    # embedding for user answer: sentence1
    sentence1 = np.array(co.embed([usr_ans]).embeddings)
    # embedding for tutor answer: sentence2
    sentence2 = np.array(co.embed([real_ans]).embeddings)

    from sklearn.metrics.pairwise import cosine_similarity
    # cosine similarity between sentence 1 and sentence 2
    rating = cosine_similarity(sentence1, sentence2)[0][0]
    if rating >= 1.3 or rating <= 0.7:
      print("Incorrect answer...")
      tutor_sadcourage = generate_text(sadprompt, temp=0.9)
      with open(sad_file_path, "w") as sad_text_file: 
      # save response as text file
        sad_text_file.write(tutor_sadcourage)

      with open(sad_file_path, "r") as sad_text_file:
        sadlines = [line for line in sad_text_file if line.strip()]
      
      if sadlines:
         random_line = random.choices(sadlines)
         print("TutorBo says: " + random_line[0])

    else:
      print("Correct answer!")
      tutor_encourage = generate_text (encourage, temp=0.9)
      with open(hype_file_path, "w") as hype_text_file: 
      # save response as text file
        hype_text_file.write(tutor_encourage)

      with open(hype_file_path, "r") as hype_text_file:
        lines = [line for line in hype_text_file if line.strip()]
      
      if lines:
         random_line = random.choices(lines)
         print("TutorBo says: " + random_line[0])

    
      



#     data = {"text1:": usr_ans, "text2": real_ans}
#     headers = {
#         "Authorization": "Bearer " + api_key
#     }
#     response = requests.post(cohere_api_endpoint, json=data, headers=headers)
#   else:
#       print("This study session is complete!")
    
# if response.status_code == 200:
#     result = response.json()
#     similarity_score = result["similarity_score"]
#     print(f"Similarity Score: {similarity_score}")
# else:
#     # print(f"API Request Failed: {response.status_code} - {response.text}")
#     print("Study session over!")

# headers = {
#     "Authorization": f"Bearer {api_key}"
# }

# cosine similarity reference: 
# https://txt.cohere.com/what-is-similarity-between-sentences/
# https://colab.research.google.com/github/cohere-ai/notebooks/blob/main/notebooks/What_Is_Similarity_Between_Sentences.ipynb?ref=txt.cohere.com#scrollTo=tZ7ls1JlkngY