import json
import openai
import requests
from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)


# create database connection
conn = sqlite3.connect('books.db', check_same_thread=False)
c = conn.cursor()

with open('my_sql_code.sql', 'r') as sql_file:
    sql_text = sql_file.read()
    c.executescript(sql_text)

conn.commit()

# Configure OpenAI API credentials
openai.api_key = 'sk-RTkqSm-y2e20zDc9ckbT3BlbkfJgwDEpqedetegp2sмet'

def extract_text_by_heading(lesson_text, heading):
    start_index = lesson_text.find(heading)
    if start_index != -1:
        start_index = lesson_text.find("\n", start_index) + 1  # Find the start of the content
        end_index = lesson_text.find("\n\n", start_index)  # Find the end of the content
        if end_index != -1:
            return lesson_text[start_index:end_index].strip()  # Extract the text and remove leading/trailing whitespace
        else:
            return lesson_text[start_index:].strip()  # If end index not found, extract until the end of the text
    else:
        return None  # Return None if the heading is not found

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_lesson_plan', methods=['POST'])
def generate_lesson_plan():
    # Retrieve form inputs
    subject = request.form['subject']
    class_level = request.form['class_level']
    learning_objectives = request.form['learning_objectives']
    duration = request.form['duration']
    teacher_name = request.form['teacher_name']

    # Generate additional sections with ChatGPT
    chatgpt_prompt = f"Subject: {subject}\nClass Level: {class_level}\nLearning Objectives: {learning_objectives} this should contain the names of sub topics \nDuration: {duration}\nTeacher Name: {teacher_name}\n\nMain Topics:\n1. Supporting material\n2. Key Vocabulary\n3. Knowledge: This refers to what the teacher wants the students to learn. It includes a list of key areas of knowledge\n4. Skills: Skills they want students to be proficient in: This includes topic-specific skills that are being developed and taken from the curriculum.\n5.Differentiation (Med): This component encourages the teacher to think about how they will make the lesson different for students who may have different learning needs.  \n6.Learning experiences (Med): This component is divided into sixsections that describe the different stages of the lesson: prepare, plan, investigate, apply, connect, and evaluate and reflect.\na) Prepare: This section of the lesson plan is focused on preparing the students for the topic that will be covered. Educators can use this time to introduce the topic, ask general questions to assess the students' prior knowledge, and engage the students with activities that will spark their interest in the topic.\nb)Plan: Fill this with the sub topics (each with a 20 word explaination) that you recommend the teacher to do in order.\nc)Investigate: During this part of the lesson, the students will be actively engaged in the topic. This might involve watching a video, conducting an experiment, or participating in a group discussion.\nd)Apply: Once the investigation is complete, the students will use the knowledge they have gained to create something. This might involve creating a poster, a presentation, or a written report.\ne)Connect: In this section, educators will help the students make connections between the topic they are studying and the world around them. This might involve discussing current events or exploring how the topic relates to different cultures or regions.\nf)Evaluate and reflect: Finally, students willreflect on what they have learned. This might include thinking about what they enjoyed, what new skills and knowledge they gained, and what they could have done better. \n7)Educator assessment: This component is focused on how the teacher will assess what the students have learned. This might involve quizzes, rubrics, or other forms of  summative end-of-lesson assessments. \n8)Educator reflection: This component encourages the teacher to reflect on the content of the lesson, whether it was at the right level, whether there were any issues, and whether the pacing was appropriate. It also encourages the teacher to reflect on whether there was enough differentiation for students with different learning needs. \n\nGenerate a indepth lesson plan with sections with headings as the main topics listed above.Dont number or put anything before the headings just put the text in a new line whenever there is a heading. You have to fill this as an educator. DONT REPLY WITH WHAT I AM GIVING YOU AS THE INPUT PLEASE. I AM DOING THIS FOR A PROJECT WHERE I WILL BE GRADED, GIVE YOUR BEST ANSWER\n"

    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=chatgpt_prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7
    )

    # Extract the generated lesson plan from ChatGPT response
    lesson_text = response.choices[0].text.strip()
    print(lesson_text)
    print('')

    headings = ['Supporting Material','Key Vocabulary','Knowledge','Skills','Differentiation ','Prepare', 'Plan', 'Investigate', 'Apply', 'Connect', 'Evaluate and Reflect', 'Educator Assessment', 'Educator Reflection']

    extracted_text = {}
    extracted_text['subject']=subject
    extracted_text['classlev']=class_level
    extracted_text['learn']=learning_objectives
    extracted_text['dura']=duration
    extracted_text['tname']=teacher_name
    for heading in headings:
        extracted_text[heading] = extract_text_by_heading(lesson_text, heading)

    with open("myfile.txt", 'w') as f:
        for key, value in extracted_text.items():
            f.write(f'{key}:{value}\n')

    c.execute("INSERT INTO books (sub,lev, learn, dura,tname,smaterial,vocab,know,skill,prep,plan,inves,apply,con,eval,eass,eref) VALUES (?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?)",
              (extracted_text['subject'], extracted_text['classlev'], extracted_text['learn'],extracted_text['dura'], extracted_text['tname'], extracted_text['Supporting Material'],extracted_text['Key Vocabulary'],extracted_text['Knowledge'],extracted_text['Skills'],extracted_text['Prepare'],extracted_text['Plan'],extracted_text['Investigate'],extracted_text['Apply'],extracted_text['Connect'],extracted_text['Evaluate and Reflect'],extracted_text['Educator Assessment'],extracted_text['Educator Reflection']))
    conn.commit()

    json_data = json.dumps(extracted_text)

    # Make a POST request to the Strapi API
    url = 'http://localhost:1337/api/books'

    headers = {
        'Authorization': 'Bearer 1cb3ac8a64d416d5f659e437267f3f53fc27e8cdf8385335057cb18d7ff1d3f63d9853e199cd7da186354c6a572e2f1e6dddc09a57cf7cc8a48810fc58287d197ebd16fe3b8648ce08cde0273a1d4467e9302199e83af7be6a14a65f097bfc68c71ee29a14d51072443fe58baac158ae27e4db44ca4b8afe123b5feaa538cc47',
        'Content-Type': 'application/json'
    }
    response = requests.post(url, data=json_data, headers=headers)

    # Handle the response from Strapi
    if response.status_code == 200:
        print('Data posted to Strapi successfully')
    else:
        print('Failed to post data to Strapi')

    return render_template('lesson_plan.html', extracted_text=extracted_text)

if __name__ == '__main__':
    app.run(debug=True)

