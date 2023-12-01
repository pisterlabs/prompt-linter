from revChatGPT.ChatGPT import Chatbot
import requests
import openai
import sqlite3
import os

# SESSION_TOKEN = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..jTdpmUzyWKhUnUl8.yYFBVmqM9kGU6XXCzzpgavw0EZsRWeZqLIQlDb1iMU-cjRZXHon7qnd9tCS15IX2mXf7bHW8r7zewe-kUcfaaSMSTBbDskNYlrBgKFt0UabAqgB-UlOxIGqE1xrv45sNDDp4p3osJbX9hfEB3n4fAa347C3PrJfzWRSGb-BRNZHjD7Ji5B48TxfHQHIoaQXfH7oG--HoPJ2j4izDJOQDwdyaU2o3WMymTjiLrhPtgstfmbKiAw7y6KNIE8JybDQvMvHGUcFAysgmK3b6sklysGX0xI8AG-Dn18Oru838pBoJQ63gKFBJU1Q8VadNIemomTv1W0tFsvXcOGrCE56mPcqV0UCN20ntcB0YueofswMIDAghbdFhg0Cy0VW4flnPJNHYSdVdmwlM_pzdPzUNX-_H76oDVSuIfEdXH5qKNyGoYD4OPaVsKr5I7sJupUvRNQfV_nivkxNIy-ekRNrcKnF2mkpCcXHma4XKQsc-FAwGVl-3fIdugjJVtYRCVaFBwrVtHiK22_KwWfkHxeMF1zWdqGS5mpxwJ7Go24Ux00CuSLB9rWt2wM9dnjojOnIYy4DqzTHhBcv_F_XoLpJQwGgNqDcR9VV5NaMMN_FHgVIlwVmb_Y2jg5VqwGYy8DdjjlCoXX4UaOEi01QQrXlimpjYdzvdQWsVv1-iabRCS7UFNPqnZmmOH-M78Obg4zTPaN4wK8qIiiZv6b9WJKBNMbYrGeujBJyi-1a5m_IX8gTaA1s0T-WjC4I1cryWrdUzlOxocROTgq-z9DAOM_MtuoLBQguqUX-Zk4Bm0bdm8Ak305p98feWyk4fy3paDGExLJ-18eWznncf8C06V5BnmeZ_E7VqUWm-SX1gR6htjRXHMQvSFwVuTvYSb09fP4DA9zqSpL7H8HACnQ8HtAaz4d_-dTMXnONQcaKgg8LL2BA8l_eGxMz5zStXMJRRxx8Ot23vSwJ5fhpbA0vnwGo6Mmmr6WBwaAiJpQ-jFhTQjqWdHAqDrmwRUNjUAA3ve0ExZnyeZpkR8DtD_8EdB56YcyQYoNxqyPS0rVvkFxg286GOb6DyVyEhXpZz_eYToFU6tOy4R1G-PHDTd5JjSy60GcbasuFFDuRDxZzUws0Pe0yfAHWwleuKOZL6vHOLPnjLgyzKhS2YlDCS0TFufiiYlc65qWqx3hEqPn6ET1n-trXUVKRceOMsJWZ1gND6jhG72MRVgl3joOooQHQ2xsF-Z1shVWhbpF4gBsF1yco0EEtVlmzQrFf0-Li_GjLhs9KSR7jc8an9dt24iHE5sBHAnp3pNPhHeXQxVlqD1AThmBD54II4o7ZgbTFBTRsKt1sEC9O6b9MQ-TqPmUpDN1BVgpRbgPLC_ZbsKylb-zjJ_CPaLyVGUcxhZAiyirsUlABC4qm7BqCTGI3JQJTj71m15C_S5hCyvmjJPOv5zKKsbH3ddgDZ7dA55zqvw-k07lTyHQBZp28OTZcQDcfkJ2sjOQjn2fN9GSqZIBNaFui4CVT9GzQO5KG5Er26PiqGG4Ls4APGgsZzK7EgVsQzOfNvYaHIK0bWP2bc1-onGlZDK4tQ1tpeDdXbigNJ1yB_k50C7J1p1o3lNZNNbC8gjNOQw32F2uUB0CEzuQM9ZzcJv8lORtQ-9nn1IF8oHr3Omj-go93px1fTZEwy4i4STUab3LiHhkD1ORU0kgowU9zwGCPPmwxe6HFaYE9dXG7mS4ddHWWBs5EjZObbKjtPR-m0cKxzeyGStt1BWULtfK6Ah3WaPl2zQS8lrxV4IBmkv0q4g-shxg6T-4FSfO1DvMngb2HfI9mFRD16uDZ63U5Ij_-Fi7GglSbG3Tw6hyimtd0IWo4G4LTldxm_EdivC-z-G4awcZve60zYa_z-GxBY1paD1oY7vw_TTcBQWF68XjdH8-bkj9N-vJdKe-NbM3LmQOyRCFFTvKNR03WIzAFGVDf9wz4svBMAvNDWsQjAuXFVgMr-rhlfVOl4i7YR_MJHZZpyDCf4luHnC5N22vOjeNg2DV2VsP4zoET1InnjLD-ple2SSyPCmA_KJjdaydDqMMA1HrHWJIfQxf2E9gjcnGBwW5aGQ8UcHsdX6wtM1UVaT5at8XzWVg454Vb1FLkAw3FtfNhELveh6-8GjsjjnZXGlRuN0Iv2Ye-yEHxDotzzqkFemWrn0v6w2IPAkiIxRmxClg6v2nDCCi7aYOxEQuv320NZMRjNsCizne1IHI5jIwLOBmmwKUOaD_He-Q1VJnL1DAM0yN9yrrH_JxI.XJWcQfEx7EOBFvK2nom3Ag'
OPENAI_KEY = os.environ.get("OPENAI_KEY") #"sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
model = "text-davinci-003"

USE_API = True

# Create the Database
conn = sqlite3.connect("./database.db", check_same_thread=False)
cur = conn.cursor()
try:
    cur.execute("CREATE TABLE QNA(id INTEGER PRIMARY KEY, question TEXT, answer TEXT)")
    conn.commit()  # Execute the statement
    cur.close()  # Close the connection
except:
    print("Table already exists, continuing...")
    pass


def getSessionToken():
    s = requests.Session()
    response = s.post()

    cookies = response.cookies
    sessionToken = cookies.get_dict()["__Secure-next-auth.session-token"]
    return sessionToken


def promptGPT3(prompt):
    print("================ [!] Using GPT3 API - Davinci ================")
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=1500,
        temperature=0,
        n=1,
    )
    return response


def askGPT(gpt_session_token, prompt, conversation_id=None, parent_id=None):
    if USE_API:
        response = promptGPT3(prompt)
    else:
        chatbot = Chatbot(
            {
                "session_token": gpt_session_token,  ## Comment out when not in use.
                # "email": "hnr.bbw@outlook.com",
                # "password": "securepassword123",
                # "isMicrosoftLogin": True
            },
            conversation_id=None,
            parent_id=None,
        )  # You can start a custom conversation

        try:
            response = chatbot.ask(
                prompt, conversation_id=conversation_id, parent_id=parent_id
            )
            return response["message"]
        except:
            response = promptGPT3(prompt)
    return response


def generateQuiz(gpt_session_token, content):
    print("received request to generate quiz...")
    instruction = "\nReturn all the questions at once first as a numbered point list, in the format Questions:\n1. Question 1\n2. Question 2\n3. Question 3\nFollow this by the answers to the question only as a numbered point list, in the format: Answers:\n1. Answer 1\n 2. Answer 2\n 2. Answer 3\n Clearly label which list is the questions list, and which list is the answers list"
    prompt = (
        "Generate 5 quiz questions and their respective answers for the following content:\n "
        + content
        + instruction
    )
    rsp = askGPT(gpt_session_token, prompt)
    print("raw gpt response", rsp)

    # Parse the response
    qna = rsp.split("\n")

    # Clean the response
    qna = [x.strip() for x in qna if x != "" and not (":" in x)]
    # qna = [x.strip() for x in qna if x != '' and x.lower() != 'questions:' and x.lower() != 'answers:']
    length = len(qna)
    questions = []
    answers = []

    # Sort data into questions and answers
    for index, qn in enumerate(qna):
        text = qn[2:] if qn[2] == " " else qn[1:]
        if index < length / 2:
            questions.append(text)
        else:
            answers.append(text)
    print(questions, answers)

    if len(questions) == len(answers):
        pass
    else:
        print("WARNING: FORMAT FOR OEA WAS NOT GIVEN CORRECTLY, ATTEMPTING TO FIX...")
        print("[DEBUG] Printing Old Questions and Answers Data: ")
        print(questions, answers)
        questions = []
        answers = []
        strippedResponse = [x.strip() for x in rsp.split("\n") if x != ""]
        try:
            for question in strippedResponse[::2]:
                questions.append(question[2:].strip())
                answers.append(
                    strippedResponse[strippedResponse.index(question) + 1]
                    .split(":")[1]
                    .strip()
                )
        except:
            print("They probably added an extra set of answers at the end :(")
            # Try to circumvent this
            try:
                badIndex = strippedResponse.index("Answers:")
            except:
                badIndex = strippedResponse.index("answers:")
            newStrippedResponse = strippedResponse[:badIndex]
            newStrippedResponse = strippedResponse[:badIndex]
            for question in newStrippedResponse[::2]:
                questions.append(question)
                answers.append(
                    newStrippedResponse[newStrippedResponse.index(question) + 1]
                )

    ## Arrange into tuple
    qna_tup = [(questions[i], answers[i]) for i in range(len(questions))]
    print(qna_tup)
    return qna_tup


def generateMCQ(gpt_session_token, content):
    print("received request to generate MCQ...")
    instruction = "\nReturn the answers in the following format:\n1. Question\nA) Option A\nB) Option B\nC) Option C\nD) Option D\n\nReturn the correct answers at the end of the response, all as one lump response on a single line, in the following format:\nAnswers: A, B, C, D, C, A"
    prompt = (
        "Generate 5 Multiple-Choice quiz questions and their respective answers for the following content:\n "
        + content
        + instruction
    )
    rsp = askGPT(gpt_session_token, prompt)
    ## Parse the response
    # Split into sections
    print(rsp)
    print("========== RESPONSE ABOVE ===========")
    sections = rsp.split("Answers:")
    if len(sections) == 1:
        sections = rsp.split("answers:")
        if len(sections) == 1:
            sections = rsp.split("ANSWERS:")
            if len(sections) == 1:
                sections = rsp.split("Answer:")
                if len(sections) == 1:
                    sections = rsp.split("answer:")
    if "question" in sections[0]:
        sections.remove(sections[0])
    if len(sections) == 2:
        print(sections)
        pass  # Normal
    else:
        print("WARNING: FORMAT FOR MCQ WAS NOT GIVEN CORRECTLY, ATTEMPTING TO FIX...")
        print(
            "Assuming Format is of form: \n1. Question\nA) Option 1\nB) Option 2\nC) Option 3\nD) Option 4\nAns: A"
        )
        lines = rsp.split("\n")
        lines = [x.strip() for x in lines if x != ""]  # clean lines
        questions = []
        possible_answers = []
        answers = []
        for question in lines[::6]:
            temp = []
            index_req = lines.index(question)
            questions.append(question[3:])
            for i in range(4):
                temp.append(lines[index_req + i + 1][3:])
            possible_answers.append(temp)
            answers.append(lines[index_req + 5].split(":")[1].strip())

            assert len(answers) == len(questions)
            print("============ QUESTIONS ===================")
            print(questions)
            print("============ POSSIBLE ANSWERS ===================")
            print(possible_answers)
            print("============ ACTUAL ANSWERS ===================")
            print(answers)
            return [
                [questions[i], possible_answers[i], answers[i]]
                for i in range(len(answers))
            ]
    qna = sections[0].split("\n")
    actualanswers = sections[1]

    ## Clean the response
    qna = [x.strip() for x in qna if x != ""]
    ## Get Questions and Possible Answers
    questions = []
    possible_answers = []
    for question in qna[::5]:
        temp = []
        index_req = qna.index(question)
        print(index_req)
        questions.append(question[3:])
        for i in range(4):
            temp.append(qna[index_req + i + 1][3:])
        possible_answers.append(temp)

    ## Get answers
    answers = actualanswers.strip().split(", ")
    assert len(answers) == len(questions)
    print("============ QUESTIONS ===================")
    print(questions)
    print("============ POSSIBLE ANSWERS ===================")
    print(possible_answers)
    print("============ ACTUAL ANSWERS ===================")
    print(actualanswers)
    return [
        [questions[i], possible_answers[i], answers[i]] for i in range(len(answers))
    ]


def generateKeyPoints(gpt_session_token, content):
    print("received request to generate Key Points...")
    instruction = "Summarize the following content into bullet points:\n"
    prompt = instruction + content
    rsp = askGPT(gpt_session_token, prompt).strip().split("\n")
    return [x[2:] for x in rsp]


def generateSummary(gpt_session_token, content):
    print("received request to generate Summary...")
    instruction = "Summarise the following content:\n"
    prompt = instruction + content
    rsp = askGPT(gpt_session_token, prompt)
    return rsp
