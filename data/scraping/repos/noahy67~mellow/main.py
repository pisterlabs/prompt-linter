import logging
import os
import psycopg2
import psycopg2.extras
from psycopg.errors import ProgrammingError
import json
from flask import Flask, jsonify, request, render_template, url_for, redirect
import UserInfo
from datetime import date
import json
import os
import cohere
import numpy as np

from dotenv import load_dotenv
load_dotenv()

#setting environment variable
conn = psycopg2.connect(os.environ["DATABASE_URL"])

with conn.cursor() as cur:
    cur.execute("SELECT * FROM disorders")
    res = cur.fetchall()
    conn.commit()
    print(res)

app = Flask(__name__)

curr_user = UserInfo.UserInfo()

# Create a cursor and initialize psycopg
pg_conn_string = os.environ["DATABASE_URL"]

connection = psycopg2.connect(pg_conn_string)

# Set to automatically commit each statement
connection.set_session(autocommit=True)

cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Retrieve all data from disorders table
def db_get_all():
    cursor.execute("SELECT * FROM disorders")
    results = cursor.fetchall()
    return results

# Retrieve all data for specialists with name 'name'
def db_get_by_name(name):
    cursor.execute('SELECT * FROM specialists WHERE name = %s', (name, ))
    result = cursor.fetchone()
    return result

# Retrieve all data for specialists covering the given disorder
def db_filter_listings(disorder):
    cursor.execute(
        'SELECT * FROM specialists WHERE disorder = %s',
        (disorder, ))
    result = cursor.fetchall()
    return result

# Create specialist entry
def db_create_specialist(disorder, name, num):
    cursor.execute(
        "INSERT INTO specialists (disorder, name, num) VALUES (%s, %s, %s)",
        (disorder, name, num))
    result = cursor.fetchall()
    return result

# Delete specialist entry
def db_delete_listing(name):
    cursor.execute("DELETE FROM specialists WHERE name = %s RETURNING name", (name, ))
    result = cursor.fetchall()
    return result

# Delete question entry
def db_delete_question(uid):
    cursor.execute("DELETE FROM questions WHERE uid = %s RETURNING uid", (uid, ))
    result = cursor.fetchall()
    return result

# Update responder_id for questions entry
def db_set_responder(uid):
    cursor.execute("UPDATE questions SET responder_uid = %s RETURNING uid", (uid, ))
    result = cursor.fetchall()
    return result

# Update questioner_id for questions entry
def db_set_questioner(uid):
    cursor.execute("UPDATE questions SET questioner_uid = %s RETURNING uid", (uid, ))
    result = cursor.fetchall()
    return result

# Retrieve all questions from given uid
def db_set_question_list(uid):
    cursor.execute("SELECT * FROM questions WHERE uid in (%s)", (uid,))
    result = cursor.fetchall()
    return result

# create questions entry
def db_create_question(uid, datey, responder_id, questioner_id, answer):
    cursor.execute(
        "INSERT INTO specialists (uid, datey, answer, questioner_id, responder_id) VALUES (%s, %s, %s, %s, %s)",
        (uid, datey, answer, questioner_id, responder_id))
    result = cursor.fetchall()
    return result

# retrieve all patients with uid
def get_user_name(uid):
    cursor.execute("SELECT name FROM patients WHERE uid in (%s)", (uid,))
    result = cursor.fetchall()
    return result

# retrieve all professionals with uid
def get_user_nameSPECIAL(uid):
    cursor.execute("SELECT name FROM professionals WHERE uid in (%s)", (uid,))
    result = cursor.fetchall()
    return result

# user-check for hacky password solution
def check_user(email):
    cursor.execute("SELECT uid FROM patients WHERE email in (%s)", (email,))
    result = cursor.fetchall()
    if not result:
        cursor.execute("SELECT uid FROM professionals WHERE email in (%s)", (email,))
        result = cursor.fetchall()
    return result[0]['uid']

# password-check for hacky password solution
def confirm_password(email, passed):
    cursor.execute("SELECT password FROM patients WHERE email in (%s)", (email,))
    result = cursor.fetchall()
    if result:
        return passed == result[0]['password']
    else:
        cursor.execute("SELECT password FROM professionals WHERE email in (%s)", (email,))
        result = cursor.fetchall()
        return passed == result[0]['password']

# retrieve questions with questioner of uid
def retrieve_user_q(uid):
    cursor.execute("SELECT * FROM questions WHERE questioner_id in (%s)", (uid,))
    return cursor.fetchall()

# retrieve questions with responder of uid
def retrieve_user_qSPECIAL(uid):
    cursor.execute("SELECT * FROM questions WHERE responder_id in (%s)", (uid,))
    return cursor.fetchall()

# checking if user is patient or pro. returns true when patient, false when pro
def check_user_type(email):
    cursor.execute("SELECT uid FROM patients WHERE email in (%s)", (email,))
    result = cursor.fetchall()
    return True if result else False

# uploads question to questions
def upload_question(question, questioner_id):
    cursor.execute("INSERT INTO questions VALUES (NULL, (%s), NULL, (%s), NULL, (%s), NULL)", (date.today(), question, questioner_id))
    return None


# given a disorder, returns the most available specialist in that field (based on num)
def match_specialist(disorder):
    print(disorder)
    cursor.execute("SELECT name FROM specialists WHERE disorder in (%s) GROUP BY name ORDER BY COUNT(num) ASC", (disorder,))

    result = cursor.fetchall()[0]['name']
    print(result)
    return result

# update responder field for questions entry
def update_responder(question, userID, responder):
    cursor.execute("UPDATE questions SET responder = (%s) WHERE questioner_id in (%s) AND question in (%s)", (responder, userID, question))
    return responder

# find responder_id for questions entry
def find_responder_id(responder):
    cursor.execute("SELECT uid FROM professionals WHERE name in (%s)", (responder,))
    responder_id = cursor.fetchall()[0]['uid']
    return responder_id

# update responder_id field for questions entry
def update_responder_id(question, userID, responder_id):
    cursor.execute("UPDATE questions SET responder_id = (%s) WHERE questioner_id in (%s) AND question in (%s)", (responder_id, userID, question))
    return None

# update answer field for questions entry
def update_answer(question, answer):
    cursor.execute("UPDATE questions SET answer = (%s) WHERE question in (%s)",(answer, question))
    return None


def signoutFUNC():
    curr_user.set_default()

def runCohere():
    Cohere_api_key = os.getenv("COHERE_API_KEY")
    names1 = ["anxiety", "depression", "schizophrenia", "bipolar", "PTSD", "eating", "neurodevelopmental"]
    descriptions = [
        "Anxiety disorders are characterised by excessive fear and worry and related behavioural disturbances. Symptoms are severe enough to result in significant distress or significant impairment in functioning. ",
        "During a depressive episode the person experiences depressed mood (feeling sad or irritable or empty) or a loss of pleasure or interest in activities for most of the day nearly every day for at least two weeks. Several other symptoms are also present which may include poor concentration or feelings of excessive guilt or low self-worth hopelessness about the future or thoughts about dying or suicide disrupted sleep or changes in appetite or weight and feeling especially tired or low in energy. ",
        "Schizophrenia is characterised by significant impairments in perception and changes in behaviour. Symptoms may include persistent delusions or hallucinations or disorganised thinking or highly disorganised behaviour or extreme agitation. People with schizophrenia may experience persistent difficulties with their cognitive functioning. Yet a range of effective treatment options exist including medication or psychoeducation or family interventions and psychosocial rehabilitation.",
        "People with bipolar disorder experience alternating depressive episodes with periods of manic symptoms.  During a depressive episode the person experiences depressed mood (feeling sad or irritable or empty) or a loss of pleasure or interest in activities for most of the day nearly every day.  Manic symptoms may include euphoria or irritability increased activity or energy and other symptoms such as increased talkativeness or racing thoughts or increased self-esteem or decreased need for sleep or distractibility and impulsive reckless behaviour."
        "PTSD may develop following exposure to an extremely threatening or horrific event or series of events.  Re-experiencing the traumatic event or events in the present (intrusive memories or flashbacks or nightmares); avoidance of thoughts and memories of the event or avoidance of activities or situations or people reminiscent of the event and  persistent perceptions of heightened current threat. A common cause of PTSD is war."
        "Disruptive behaviour and dissocial disorders are characterised by persistent behaviour problems such as persistently defiant or disobedient to behaviours that persistently violate the basic rights of others or major age-appropriate societal norms or rules or laws. Onset of disruptive and dissocial disorders is commonly though not always during childhood. Another common symptom is not eating or starving yourself because you don't like how you look."
        "Neurodevelopmental disorders include ADHD and ASD. ADHD is characterised by a persistent pattern of inattention and/or hyperactivity-impulsivity that has a direct negative impact on academic occupational or social functioning.  Disorders of intellectual development are characterised by significant limitations in intellectual functioning and adaptive behaviour which refers to difficulties with everyday conceptual or social and practical skills that are performed in daily life. Autism spectrum disorder (ASD) constitutes a diverse group of conditions characterised by some degree of difficulty with social communication and reciprocal social interaction as well as persistent restricted repetitive and inflexible patterns of behaviour interests or activities."
    ]

    co = cohere.Client(Cohere_api_key)
    response = co.embed(texts=descriptions)
    embeddings = np.array(response.embeddings)

    with open("embeddings.npy", "wb") as f:
        np.save(f, embeddings)

    with open("descriptions.json", "w") as f:
        json.dump(descriptions, f)

    with open("names1.json", "w") as f:
        json.dump(names1, f)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/profileConsumer")
def profileConsumer1():

    curr_user.name = get_user_name(curr_user.userID)[0]['name']
    name = curr_user.name
    curr_user.questions = retrieve_user_q(curr_user.userID)
    questions = curr_user.questions
    print(questions)
    logged = curr_user.loggedIn
    numQuestions = 0
    if questions:
        numQuestions = len(questions)
    return render_template('profileConsumer.html', name=name, logged=logged, questions=questions, numQuestions=numQuestions, type=curr_user.type)

@app.route("/profileConsumer",  methods=['POST', 'GET'])
def profileConsumerFuncs():
    name = curr_user.name
    questions = curr_user.questions
    best_disorder = ""
    numQuestions = 0
    if questions:
        numQuestions = len(questions)
    logged = curr_user.loggedIn
    if 'questionSubmit' in request.form:
        question = request.form['question']
        # PUT THE QUESTION IN THE DATABASE AND RELOAD THE PAGE
        upload_question(question, curr_user.userID)


        runCohere()
        Cohere_api_key = os.getenv("COHERE_API_KEY")

        def cosine_distance(A, B):
            return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

        with open("embeddings.npy", "rb") as f:
            embeddings = np.load(f)

        with open("descriptions.json", "r") as f:
            descriptions = json.load(f)

        with open("names1.json", "r") as f:
            names1 = json.load(f)

        co = cohere.Client(Cohere_api_key)
        response = co.embed(texts=[question])
        question_embedding = np.array(response.embeddings[0])

        best_disorder = None
        best_score = -1000
        for name1, embedding in zip(names1, embeddings):
            score = cosine_distance(embedding, question_embedding)
            if score > best_score:
                best_disorder = name1
                best_score = score

        # FUNCTION TO GIVE TO EXPERT

        responder = match_specialist(best_disorder)
        responder = update_responder(question, curr_user.userID, responder)
        responder_id = find_responder_id(responder)
        update_responder_id(question, curr_user.userID, responder_id)


        curr_user.questions = retrieve_user_q(curr_user.userID)
        questions = curr_user.questions
        numQuestions = 0
        if questions:
            numQuestions = len(questions)

        return render_template('profileConsumer.html', name=name, logged=logged, questions=questions, numQuestions=numQuestions, type=curr_user.type, disorder=best_disorder)
    return render_template('profileConsumer.html', name=name, logged=logged, questions=questions, numQuestions=numQuestions, type=curr_user.type, disorder=best_disorder)


@app.route("/profileProfessional")
def profileProfessional1():
    curr_user.name = get_user_nameSPECIAL(curr_user.userID)[0]['name']
    name = curr_user.name
    curr_user.questions = retrieve_user_qSPECIAL(curr_user.userID)
    questions = curr_user.questions
    numQuestions = 0
    qNAME = "Asker:"
    if questions:
        numQuestions = len(questions)
        qNAME = "Asker:"
    logged = curr_user.loggedIn

    return render_template('profileProfessional.html', name=name, logged=logged, questions=questions, numQuestions=numQuestions, type=curr_user.type, qNAME=qNAME)


@app.route("/profileProfessional", methods=['POST', 'GET'])
def profileProfessionalFuncs():
    name = curr_user.name
    curr_user.questions = retrieve_user_qSPECIAL(curr_user.userID)
    questions = curr_user.questions
    numQuestions = 0
    qNAME = "Asker:"
    if questions:
        numQuestions = len(questions)
        qNAME = "Asker:"
    logged = curr_user.loggedIn
    if 'submitAnswer' in request.form:
        answer = request.form['answer']
        update_answer(questions[0]['question'], answer)
        # UPDATE THE QUESTION ANSWER HERE

    return render_template('profileProfessional.html', name=name, logged=logged, questions=questions, numQuestions=numQuestions, type=curr_user.type, qNAME=qNAME)


@app.route('/signin')
def signin():
    return render_template('signin.html', title='Signin')


@app.route('/signin', methods=['POST', 'GET'])
def signinFuncs():
    if 'signin' in request.form:
        email = request.form['email']
        password = request.form['password']
        status = 'yes'
        # SIGN USER IN

        real = confirm_password(email, password)
        print(real)
        if not real:
            return render_template('signin.html', title='Signin', status='no')
        curr_user.set_email(email)
        curr_user.set_log_status('true')

        curr_user.userID = check_user(email)
        curr_user.type = check_user_type(email)

        # GET USER INFORMATION AND WHETHER THEY ARE Prof or MAT
        if curr_user.type:
            return redirect(url_for('profileConsumer1'))
        return redirect(url_for('profileProfessional1'))
    return render_template('signin.html', title='Signin')


@app.route('/signupConsumer')
def signupConsumer():
    return render_template('signupConsumer.html', title='Signup Consumer')


@app.route('/signupConsumer', methods=['POST', 'GET'])
def signupConsumerFuncs():
    if 'signup' in request.form:
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirmpassword']
        first = request.form['firstname']
        last = request.form['lastname']

        # if password != confirm:
        #     return render_template('signup.html', title='Signup', status='diff')

        # result = storage.signupUser(email, password, first, last)
        # if result == 's':
        #     curr_user.set_user(email)
        #     curr_user.set_log_status('true')
        #     return redirect(url_for('index'))
        # if result == 'us':
        #     return render_template('signup.html', title='Signup', status='no')

        # WRITE SIGNUP TO DATABASE

    return render_template('signupConsumer.html', title='Signup Consumer')



@app.route('/signupProfessional')
def signupProfessional():
    return render_template('signupProfessional.html', title='Signup Professional')


@app.route('/signupProfessional', methods=['POST', 'GET'])
def signupProfessionalFuncs():
    if 'signup' in request.form:
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirmpassword']
        first = request.form['firstname']
        last = request.form['lastname']
        job = request.form['profession']
        # if password != confirm:
        #     return render_template('signup.html', title='Signup', status='diff')

        # result = storage.signupUser(email, password, first, last)
        # if result == 's':
        #     curr_user.set_user(email)
        #     curr_user.set_log_status('true')
        #     return redirect(url_for('index'))
        # if result == 'us':
        #     return render_template('signup.html', title='Signup', status='no')

        # WRITE SIGNUP TO DATABASE // TODO
    return render_template('signupProfessional.html', title='Signup Professional')

if __name__ == '__main__':
    app.run(debug=False)