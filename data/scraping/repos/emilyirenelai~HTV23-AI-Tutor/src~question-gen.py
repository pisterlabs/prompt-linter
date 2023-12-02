# must install cohere in terminal to use:
# python -m pip install cohere 


import cohere
import fnmatch
import random
co = cohere.Client('${{ secrets.COHERE_KEY }}')
api_key = '${{ secrets.API_KEY }}'

def generate_text(prompt, temp=0):
  response = co.generate(
    model='command',
    prompt=prompt,
    max_tokens=200,
    temperature=temp)
  return response.generations[0].text

# context will be replaced with user input prompt
context = "When it comes to designing software, effective design is very important. Connecting to databases can be an extremely expensive task, particularly because of how much time it typically takes. The inefficient way to use a database connection would be connecting and disconnecting multiple times every time the database is required. Because of this, the best practice is to connect to it once, perform all the necessary functions dealing with the database, and then disconnect. Such a practice can be followed by using the singleton pattern in database connections. This involves setting up the database only once. Then, whichever methods require calling the database will leverage this connection. This type of design is efficient in multiple ways. The primary business concern would be that it is fast, meaning that it spends less company time and is therefore less expensive. It is also beneficial for the environment, as it saves energy and processing power. The reason for this is that excessive connections and disconnections to the database are avoided."
prompt = f"""Write questions based on this lesson: {context}, giving one question for every main topic. Write an answer on the following line. The format should look like:
Question:
Answer:

"""

# Example format:
# Input Lesson: Engineers, as practitioners of engineering, are professionals who invent, design, analyze, build and test machines, complex systems, structures, gadgets and materials to fulfill functional objectives and requirements while considering the limitations imposed by practicality, regulation, safety and cost.[1][2] The word engineer (Latin ingeniator[3]) is derived from the Latin words ingeniare ("to contrive, devise") and ingenium ("cleverness").[4][5] The foundational qualifications of a licensed professional engineer typically include a four-year bachelor's degree in an engineering discipline, or in some jurisdictions, a master's degree in an engineering discipline plus four to six years of peer-reviewed professional practice (culminating in a project report or thesis) and passage of engineering board examinations.
# The work of engineers forms the link between scientific discoveries and their subsequent applications to human and business needs and quality of life.Engineers develop new technological solutions. During the engineering design process, the responsibilities of the engineer may include defining problems, conducting and narrowing research, analyzing criteria, finding and analyzing solutions, and making decisions. Much of an engineer's time is spent on researching, locating, applying, and transferring information.[7] Indeed, research suggests engineers spend 56% of their time engaged in various information behaviours, including 14% actively searching for information.
# Engineers must weigh different design choices on their merits and choose the solution that best matches the requirements and needs. Their crucial and unique task is to identify, understand, and interpret the constraints on a design in order to produce a successful result. 
# Questions:
# Question 1: What are three things that engineers do?
# Answer 1: Invent, design, build.
# Question 2: What Latin words did the word "engineer" originate from?
# Answer 2: The word engineer originated from the Latin words ingeniare ("to contrive, devise") and ingenium ("cleverness").
# Question 3: What level of education are engineers expected to have?
# Answer 3: The foundational qualifications of a licensed professional engineer typically include a four-year bachelor's degree in an engineering discipline, or in some jurisdictions, a master's degree in an engineering discipline plus four to six years of peer-reviewed professional practice (culminating in a project report or thesis) and passage of engineering board examinations.
# Question 4: Why do engineers need to identify constraints?
# Answer 4: Engineers need to identify constraints in order to produce a successful result for their design.
# Question 5: How do engineers spend most of their design time?
# Answer 5: Engineers spend most of their time researching, locating, applying, and transferring information.
# Question 6: List three responsibilities of an engineer during the engineering design process.
# Answer 6: Defining problems, conducting and narrowing research, analyzing criteria, finding and analyzing solutions, making decisions.

question_response = generate_text (prompt, temp=0.5)
# print("ORIGINAL QUESTIONS AND ANSWERS:")
# print(question_response)

# file path for questions
file_path = "src/qs/question.txt"

with open(file_path, "w") as q_text_file: 
  # save response as text file
  q_text_file.write(question_response)

# / test for printing the text file to ensure it comes out correctly
# with open(file_path, "r") as q_txt:
  # / read and print each line
  # print("FILE PRINT:")
  # for line in q_txt:
    # print(line, end="")

# print(question_response)

# / take this response and instead of printing it
# / first line (question) becomes output for viewer to see
# / counting function:
# with open("src/qs/question.txt", 'r') as fp:
# 	for count, line in enumerate(fp):
# 		pass
# count for number of lines in text file     
# line_counter = count+1
q_s=[]
a_s=[]

with open("src/qs/question.txt", 'r') as file:
     for line in file:
          line = line.strip()

          if line.startswith("Question"):
               q_s.append(line)
          elif line.startswith("Answer"):
               a_s.append(line)

print(q_s)
print(a_s)
# now we should have an array of questions q_s
# and an array of answers a_s          

# second line is answer that user's input is compared against
# loop continue calling the question-making function. end when 


# above this line is where we get the questions and answers arrays
#####################################################################
# below this line is the interactions: this could also be moved to js
# what this does is ask you a question, then prompt you to answer.
# then once you answer it sends a message of encouragemeny
# 

import requests

# Replace with your Cohere API endpoint and API key
cohere_api_endpoint = "https://api.cohere.ai/v1/embed"
api_key = "L2VMOXwleskZQjVuP5QEe2puJTKNLAzGaRhSEVTK"

# output question
counter = len(q_s)
encourage = "Write a short encouraging phrase 2-10 words long to encourage a student to keep going! Be creative."
hype_file_path = "src/qs/hype.txt"

for i in range(counter):
  if q_s[i] and a_s[i]:
    print(q_s[i])
    usr_ans = input("Answer TutorBo! ")
    real_ans = (a_s[i])
    tutor_encourage = generate_text (encourage, temp=0.9)

    with open(hype_file_path, "w") as hype_text_file: 
      # save response as text file
      hype_text_file.write(tutor_encourage)

    with open(hype_file_path, "r") as hype_text_file:
      lines = [line for line in hype_text_file if line.strip()]
      
      if lines:
         random_line = random.choices(lines)
         print("TutorBo says: " + random_line[0])

      # for i, line in enumerate(hype_text_file, 1):
      #     if i == random.randint(1,3):
      #         print(line)
      # print(hype_text_file)

    #print (tutor_encourage)

    data = {"text1:": usr_ans, "text2": real_ans}
    headers = {
        "Authorization": "Bearer " + api_key
    }
    response = requests.post(cohere_api_endpoint, json=data, headers=headers)
  else:
      print("This study session is complete!")
    
if response.status_code == 200:
    result = response.json()
    similarity_score = result["similarity_score"]
    print(f"Similarity Score: {similarity_score}")
else:
    # print(f"API Request Failed: {response.status_code} - {response.text}")
    print("Study session over!")

headers = {
    "Authorization": f"Bearer {api_key}"
}

# Make the API request
# response = requests.post(cohere_api_endpoint, json=data, headers=headers)

# if response.status_code == 200:
#     result = response.json()
#     similarity_score = result["similarity_score"]
#     print(f"Similarity Score: {similarity_score}")
# else:
#     print(f"API Request Failed: {response.status_code} - {response.text}")

