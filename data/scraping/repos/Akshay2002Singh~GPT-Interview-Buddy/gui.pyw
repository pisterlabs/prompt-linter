from tkinter import *
from tkinter import messagebox
from PIL import ImageTk,Image
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import speech_recognition as sr
import queue
import threading
from gtts import gTTS
from playsound import playsound
import os
import time
import json
import speedtest
import webbrowser

# some global variables
roles = ["Python Developer","Frontend Developer","Django Developer","React Developer","MERN Developer","Android Developer","MEAN Developer","Software Developer","Java Developer","Ethical Hacker","Game Developer", " Network Engineer","Database Administrator"," Machine Learning Engineer"]
current_question = 0
questions = ["Tell me something about Yourself?"]
answer_feedback = {}
user_score_obj = {}
user_answer_obj = {}
user_score = 0
# queue to store speech recognition object and speech clips
speech_queue = queue.SimpleQueue()


# function to bring general questions
def get_general_questions(attempt=0):
    global questions
    # create model to predict
    if attempt > 1:
        return
    try:
        llm = OpenAI(openai_api_key=apiKey.get())
        # Prompt template for getting questions
        prompt_search_questions = PromptTemplate.from_template("Provide minimum 6 interview questions based on dsa and basic coding for {experience} candidate?")
        # format template to final prompt
        llm_questions = prompt_search_questions.format(experience = experience.get())
        # Getting output of prompt 
        questions_output = llm.predict(llm_questions)

        # convert questions from string to list
        questions_output = questions_output.strip()
        questions_output = questions_output.split("\n")
        questions_lst = []
        for i in questions_output:
            i = i.strip()
            try:
                i = i.split('. ')
                questions_lst.append(i[1])
            except:
                pass
        print(questions_lst)
        
        # update question list 
        questions = questions + questions_lst
        # add some dsa and reasoning questions
    except:
        get_rolespecific_questions(attempt+1)

# function to bring role specific questions
def get_rolespecific_questions(attempt=0):
    global questions
    # create model to predict
    if attempt > 2:
        return
    try:
        llm = OpenAI(openai_api_key=apiKey.get())
        # Prompt template for getting questions
        prompt_search_questions = PromptTemplate.from_template("Provide minimum 13 interview questions for {role} for {experience} candidate?")
        # format template to final prompt
        llm_questions = prompt_search_questions.format(role = role.get(), experience = experience.get())
        # Getting output of prompt 
        questions_output = llm.predict(llm_questions)

        # convert questions from string to list
        questions_output = questions_output.strip()
        questions_output = questions_output.split("\n")
        questions_lst = []
        for i in questions_output:
            i = i.split('. ')
            questions_lst.append(i[1])
        print(questions_lst)
        # remove check_internet_mic => instrction screeen 
        check_internet_mic.forget()
        # update question list 
        questions = questions + questions_lst
        # add some dsa and reasoning questions

        # show new frame for question page 
        f2.pack(fill=BOTH)
        # call next question function to render question
        update_status("Ready to go")
        next_question()
    except:
        get_rolespecific_questions(attempt+1)

# this function convert text to speech 
def text_to_speech(text, try_count = 0):
    if try_count > 6:
        return
    try:
        speech = gTTS(text = text, lang='en', tld='co.in')
        speech.save("temp.mp3")
        # play it 
        playsound('temp.mp3')
        os.remove("temp.mp3")
    except:
        time.sleep(0.2)
        text_to_speech(text, try_count+1)

def score_answer(question_number, answer, attempt=0):
    global user_score
    global answer_feedback
    global user_score_obj
    if attempt > 3:
        return
    
    try:
        # create model to predict
        chat = ChatOpenAI(openai_api_key=apiKey.get())

        # Prompt templates to get score of answer
        system_template = (
            '''
            You are an interviewer, taking interview for {role} of a {experience} candidate.
            Question : {question}
            '''
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = "Answer : {answer}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        system_template_1 = "Rate this answer from 0 to 10 and provide score and feedback in json format."
        system_message_prompt_1 = SystemMessagePromptTemplate.from_template(system_template_1)
        # create final chat prompt 
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, system_message_prompt_1]
        )
        # Getting output of prompt 
        score_feedback = chat(
            chat_prompt.format_prompt(
                role = role.get(), experience=experience.get(), question=questions[question_number], answer = answer
            ).to_messages()
        )
        score_feedback = json.loads(score_feedback.content)
        user_score += int(score_feedback["score"])
        answer_feedback[question_number] = score_feedback['feedback']
        user_score_obj[question_number] = score_feedback['score']
        user_answer_obj[question_number] = answer
    except:
        score_answer(question_number,answer,attempt+1)
    

# this function read queue, if it has any clips then convert it to text
def process_speech_to_text_queue():
    # print("in queue processing function")
    while(record_users_answer.get() or speech_queue.qsize() >0):
        if(speech_queue.qsize() > 0):
            # print(f"in queue {speech_queue.qsize()}")
            # get speech recognition object and audio clip
            object_audio = speech_queue.get()
            try:
                text = ""
                # convert clip to text 
                text = object_audio[0].recognize_google(object_audio[1])
                user_answer.set(user_answer.get() + text +". ")
            except:
                # print("Something went wrong")
                pass

        time.sleep(0.5)
    next_question()


# this code records audio 
def record_audio():
    # print("in record function")
    while(record_users_answer.get()):
        # print(record_users_answer.get())
        try:
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                # Initialize the recognizer
                r = sr.Recognizer()
                # wait for a second to let the recognizer adjust the energy threshold based on the surrounding noiselevel
                r.adjust_for_ambient_noise(source2, duration=0.1)
                # listens for the user's input
                audio2 = r.listen(source2)
                # put recognizer object and audio clip in queue 
                speech_queue.put([r,audio2])
                # print(f'inside record = {speech_queue.qsize()}')
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            print("unknown error occurred")

# this function render new question
def next_question():
    global current_question
    global consider_textarea_answer
    update_status("Get Ready for question")
    if(consider_textarea_answer.get()):
        user_answer.set(textarea.get(1.0,END))
    if current_question <= len(questions) and current_question > 0:
        threading.Thread(target=score_answer, args=[current_question-1,user_answer.get(),]).start()
    
    # print(user_answer.get())
    user_answer.set("")
    textarea.delete(1.0,END)
    f2_checkbox["state"] = "normal"
    consider_textarea_answer.set(False)
    if current_question<len(questions):
        # hide mic and next btn 
        mic_lable.forget()
        next_button.forget()
        space1.forget()
        recording_btn.forget()

        user_question.set(f'Question : {questions[current_question]}')
        text_to_speech(questions[current_question])
        current_question+=1

        # show mic and next btn 
        mic_lable.pack(ipady=15)
        mic_lable.bind("<Button-1>", mic_clicked)
        recording_btn.pack()
        recording_btn['state'] = "normal"
        space1.pack()
        next_button.pack()
        next_button["state"] = "normal"

        update_status("Waiting for your response")
    else:
        show_result()

def download_report():
    questions_cards = ''
    for i in range(len(questions)):
        questions_cards += f'''
        <div class="question_card">
                <div class="question"><span>Question {i+1}:</span>  {questions[i]}</div>
                <div class="score"><span>Score:</span> {user_score_obj[i]}/10</div>
                <div class"feedback"><span>User Answer:</span> {user_answer_obj[i]}</div>
                <div class"feedback"><span>Feedback:</span> {answer_feedback[i]}</div>
            </div>
    ''' 
        
    report = '''
        <!DOCTYPE html>
        <html lang="en">

        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Report Page</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Abyssinica+SIL&family=Montserrat&family=Poppins:wght@300;400;500&display=swap');

                * {
                    margin: 0px;
                    padding: 0px;
                    box-sizing: border-box;
                    font-family: 'Poppins', sans-serif;
                }
                body {
                    background-color: #171717;
                }
                .text-center {
                    text-align: center;
                }
                .container {
                    max-width: 1100px;
                    margin: auto;
                    background-color: rgb(92, 213, 209);
                    padding: 10px 20px;
                }
                .top-bar {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    margin: 10px 0px;
                    font-size: 20px;
                    font-weight: 500;
                }
                .top-bar span{
                    font-weight: 600;
                }
                .mr-20 {
                    padding-right: 20px;
                }
                #ques_container {
                    margin-top: 20px;
                }
                .question_card {
                    margin-bottom: 30px;
                    font-size: 18px;
                }
                .question_card span{
                    font-weight: 500;
                }
            </style>
        </head>
        '''

    report += f'''
        <body>
            <div class="container">
                <h1 class="text-center">GPT Interview Buddy</h1>
                <h2 class="text-center">Total Score: {user_score}/{10*len(questions)}</h2>
                <div class="top-bar">
                    <div class="mr-20"><span>Role:</span> {role.get()}</div>
                    <div><span>Experience:</span> {experience.get()}</div>
                </div>
                <div id="ques_container">
                    {questions_cards}
                </div>
                <div class="text-center">Created by Rapid Coders</div>
            </div>
        </body>

        </html>
    '''

    with open("User_Report.html",'w') as f:
        f.write(report)
    messagebox.showinfo("Report Downloaded", "Your Report is downloaded as 'User_Report' in current folder")


def show_result():
    update_status("Wait until your report is getting ready")
    while(len(answer_feedback.keys()) != len(questions)):
        time.sleep(1)
    update_status("Your Report is ready")
    global current_question
    current_question = 0
    f2.forget()
    f3.pack()
    resultPage_total_score.set(f"Your Total Score : {user_score}/{len(questions)*10}")
    resultPage_current_score.set(f"Current Question Score : {user_score_obj[current_question]}/10")
    resultPage_question.set(f"Question {current_question+1} : {questions[current_question]}")
    resultPage_feedback.set(f"Feedback : {answer_feedback[current_question]}")
    feedback_previous_btn['state'] = 'disable'

def previous_feedback():
    global current_question
    if feedback_next_btn['state'] == 'disabled':
        feedback_next_btn['state'] = 'normal'
    current_question -= 1
    resultPage_current_score.set(f"Current Question Score : {user_score_obj[current_question]}/10")
    resultPage_question.set(f"Question {current_question+1} : {questions[current_question]}")
    resultPage_feedback.set(f"Feedback : {answer_feedback[current_question]}")
    if(current_question == 0):
        feedback_previous_btn['state'] = 'disabled'

def next_feedback():
    global current_question
    if feedback_previous_btn['state'] == 'disabled':
        feedback_previous_btn['state'] = 'normal'
    current_question += 1
    resultPage_current_score.set(f"Current Question Score : {user_score_obj[current_question]}/10")
    resultPage_question.set(f"Question {current_question+1} : {questions[current_question]}")
    resultPage_feedback.set(f"Feedback : {answer_feedback[current_question]}")
    if( current_question == (len(questions)-1) ):
        feedback_next_btn['state'] = 'disabled'

# check_current answer and show next question
def check_and_call_next():
    threading.Thread(target=next_question).start()

# this function handle mic click 
def mic_clicked(temp = None):
    if recording_btn.cget('text') == "Start Recording":
        update_status("Recording...")
        # set flag to true, to start recording 
        record_users_answer.set(True)
        recording_btn.config(text="Stop Recording", background='#d93b3b')
        # disable next btn and consider text area checkbox
        next_button["state"] = "disabled"
        f2_checkbox["state"] = "disabled"
        # call record and convert functions
        threading.Thread(target=record_audio).start()
        threading.Thread(target=process_speech_to_text_queue).start()
    else:
        update_status("Evaluating your answer")
        recording_btn.config(text="Start Recording", background='#2c70e6')
        record_users_answer.set(False)
        recording_btn['state'] = "disabled"
        mic_lable.unbind("<Button-1>")

# check whether api key is valid or not 
def is_api_key_valid():
    update_status("Checking API Key")
    try:
        llm = OpenAI(openai_api_key=apiKey.get())
        response = llm.predict("Are you ok")
    except:
        return False
    else:
        return True

# function to start interview 
def start_interview():
    if(apiKey.get().strip() == ""):
        update_status("API key is not provided")
        return
    # calling this function with thread to avoid freezing of screen
    start_button["state"] = "disabled"
    api_key_entry.configure(state="disabled")
    update_status("Please wait, checking internet and microphone (this may take 2-3 min)")
    threading.Thread(target=check_speed_mic).start()
    threading.Thread(target=get_general_questions).start()

def move_forward():
    update_status("Please wait, finding the right interviewer for you")
    threading.Thread(target=get_rolespecific_questions).start()

def checkMic():
    try:
        obj = sr.Microphone()
    except:
        return False
    return True

def check_internet(attempt=0):
    if(attempt>1):
        return 0
    try:
        st = speedtest.Speedtest(secure=True)
        speed = st.download()/(1024*1024)
        speed = str(speed)
        speed = speed[:speed.find(".")+2]
        speed = float(speed)
        return speed
    except:
        return check_internet(attempt+1)

def check_speed_mic():
    download = check_internet()
    mic_working = checkMic()

    created_by.forget()
    f1.forget()

    check_internet_mic.pack()
    Label(check_internet_mic,text=f"Your download speed is {download} Mbps",font="calibre 16 bold",fg="black").pack(ipady=5)
    if(mic_working):
        Label(check_internet_mic,text="Your microphone is working properly",font="calibre 16 bold",fg="black").pack(ipady=5)
    else:
        Label(check_internet_mic,text="Your default microphone is not available, check the system settings",font="calibre 16 bold",fg="black").pack(ipady=5,)
        update_status("Come back later after fixing mic")
        return
    
    if(download<1):
        update_status("Your internet connection is either dead or your speed is too low")
        return
    
    if(is_api_key_valid() == False):
        update_status("Invalid API key")
        return
    
    Label(check_internet_mic,text="Instructions",font="calibre 20 bold",fg="black").pack(ipady=5)
    Label(check_internet_mic,text="1.) You can start giving answer after the interviewer speak question",font="calibre 15 bold",fg="black",wraplength=1100).pack(ipady=3)
    Label(check_internet_mic,text="2.) You can give your answer either by speaking or by writing",font="calibre 15 bold",fg="black",wraplength=1100).pack(ipady=3)
    Label(check_internet_mic,text="3.) Use textarea to write answer only when needed",font="calibre 15 bold",fg="black",wraplength=1100).pack(ipady=3)
    Label(check_internet_mic,text="4.) When you are using the text area to give an answer, tick the checkbox",font="calibre 15 bold",fg="black",wraplength=1100).pack(ipady=3)
    Label(check_internet_mic,text="",font="calibre 15 bold").pack()
    
    Button(check_internet_mic,text="Move Forward",command=move_forward,font="calibre 17 bold",width=18,background='#b3fbfc').pack()
    update_status("You are ready to go")
    
def update_status(msg):
    status.set(f"Status : {msg}")

# function to open browser
def callback(url):
   webbrowser.open_new_tab(url)

if __name__ == "__main__":
    root = Tk()
    # setup basic window
    root.title("GPT Interview Buddy")
    root.geometry("1200x780")
    root.minsize(1150,740)

    # define Variables 
    role = StringVar(value=roles[0])
    experience = StringVar(value="Fresher")
    apiKey = StringVar(value="")
    user_question = StringVar(value="nothing")
    user_answer = StringVar(value="")
    record_users_answer = BooleanVar(value=False)
    consider_textarea_answer = BooleanVar(value=False)
    status = StringVar(value="Status : Ready to go")
    # Variable of result page 
    resultPage_current_score = StringVar(value="")
    resultPage_total_score = StringVar(value="")
    resultPage_question = StringVar(value="")
    resultPage_feedback = StringVar(value="")

    # Main headings 
    Label(root,text="GPT Interview Buddy",font="calibre 25 bold").pack()
    created_by = Label(root,text="Created By Rapid Coders",font="calibre 15 normal",fg="#ff0066")
    created_by.pack()
    Label(root,text="",font="calibre 2 bold").pack()

    # Creating frame to hold all content of home page
    f1 = Frame(root)
    f1.pack(fill=BOTH)

    # Inserting options to select role
    # frame 1 is home page 
    Label(f1,text="Select Your Role",font="calibre 20 bold",fg="black",bg='#bcecf5', relief='sunken' , pady=2).pack(side=TOP, ipady=5, ipadx=8)
    i = 0
    while(i<len(roles)):
        temp_frame = Frame(f1)
        temp_frame.pack(side=TOP,fill=Y)
        Radiobutton(temp_frame,text=roles[i], font="cosmicsansms 15", width=20 , padx=15,variable=role,value=roles[i]).pack(side=LEFT)
        i+=1
        if(i<len(roles)):
            Radiobutton(temp_frame,text=roles[i], font="cosmicsansms 15", width=20, padx=15,variable=role,value=roles[i]).pack(side=RIGHT)
            i+=1
            
    # Creating frame to insert experience options
    temp_frame = Frame(f1, pady=10)
    temp_frame.pack(fill=Y)
    Label(temp_frame,text="Select Your Experience",font="calibre 20 bold",fg="black" ,bg='#bcecf5', relief='sunken', pady=2).pack(side=TOP, ipady=5, ipadx=8)
    Radiobutton(temp_frame,text='Fresher', font="cosmicsansms 15", width=20 , padx=15, variable=experience, value="Fresher").pack(side=LEFT, ipady=10)
    Radiobutton(temp_frame,text='Intermediate', font="cosmicsansms 15", width=20 , padx=15, variable=experience, value="Intermediate").pack(side=LEFT, ipady=10)
    Radiobutton(temp_frame,text='Senior', font="cosmicsansms 15", width=20 , padx=15, variable=experience, value="Senior").pack(side=LEFT, ipady=10)

    api_key_frame = Frame(f1)
    api_key_frame.pack(side=TOP,ipady=10)
    Label(api_key_frame,text="Enter OpenAI API Key : ",font="cosmicsansms 18").pack(side=LEFT)
    api_key_entry = Entry(api_key_frame,textvariable=apiKey,font="cosmicsansms 18")
    api_key_entry.pack(side=LEFT)
    #Create a Label to display the link
    api_link = Label(f1, text="Don't have API Key? Click here",font=('Helveticabold', 15), fg="blue", cursor="hand2")
    api_link.pack()
    api_link.bind("<Button-1>", lambda e: callback("https://platform.openai.com/account/api-keys"))
    # Button to start interview 
    Label(f1,text="",font="cosmicsansms 10").pack()
    start_button = Button(f1,text="Start Interview",command=start_interview,font="calibre 17 bold")
    start_button.pack()

    # check and show internet and mic status and show instructions
    check_internet_mic = Frame(root)

    # this will be rendered later
    # frame 2 is second page(question page)
    f2 = Frame(root)
    # lable to show question
    Label(f2,textvariable=user_question,padx=15,font="calibre 15 normal",wraplength=1100).pack()
    # show input box as an option to write answer
    textarea_frame = Frame(f2)
    textarea_frame.pack(ipady=8)
    textarea=Text(textarea_frame,font=("Ariel",14,"normal"),height=7,width=70)
    textarea.pack(side=LEFT)
    # adding Scrollbar to textarea
    Scroll =Scrollbar(textarea_frame)
    Scroll.pack(side=RIGHT,fill=Y)
    Scroll.config(command=textarea.yview)
    textarea.config(yscrollcommand=Scroll.set)
    # show check box to switch between text area and mic 
    # if checkbox is ticked then answer written in text area will be considered
    f2_checkbox = Checkbutton(f2,text="Consider the answer given in textarea instead of audio",variable=consider_textarea_answer,font="calibre 15 normal")
    f2_checkbox.pack()
    # show mic image and set onclick event
    mic = ImageTk.PhotoImage(Image.open('mic.png'), height=20, width= 20)
    mic_lable = Label(f2, image=mic)
    mic_lable.bind("<Button-1>", mic_clicked)
    recording_btn = Button(f2,text="Start Recording", command=mic_clicked, font="calibre 17 bold",borderwidth=2, background='#2c70e6', fg='#edfa00')
    space1 = Label(f2,text="")
    next_button = Button(f2,text="Next Question",command=check_and_call_next,font="calibre 17 bold")

    # this page will be rendered later
    # frame 3 is third page(Result page)
    f3 = Frame(root)
    Label(f3,textvariable=resultPage_total_score,font="calibre 22 bold").pack(pady=5)
    Label(f3,textvariable=resultPage_question, font="calibre 17 bold" ,wraplength=1100).pack(pady=5)
    Label(f3,textvariable=resultPage_current_score,font="calibre 14 bold").pack()
    Label(f3,textvariable=resultPage_feedback,font="calibre 14 normal",wraplength=1100).pack(pady=5)
    temp_f3 = Frame(f3)
    temp_f3.pack(pady=20)
    feedback_previous_btn =  Button(temp_f3,text="Previous Question",command=previous_feedback,font="calibre 17 bold",width=18,background='#b3fbfc')
    feedback_previous_btn.pack(side=LEFT)
    Label(temp_f3,text="\t \t \t \t").pack(side=LEFT)
    feedback_next_btn = Button(temp_f3,text="Next Question",command=next_feedback,font="calibre 17 bold",width=18,background='#b3fbfc')
    feedback_next_btn.pack(side=RIGHT)

    Button(f3,text="Download Report",command=download_report,font="calibre 17 bold",width=20,background='#f3b5ff').pack(side=TOP)
    Label(f3,text="").pack(pady=2)
    Button(f3,text="Quit",command=quit,font="calibre 17 bold",width=20,bg='#f3b5ff').pack(side=TOP)

    # status bar 
    status_frame = Frame(root,bg="#7ae9fa",relief='solid',border=2)
    status_frame.pack(side=BOTTOM,fill=X)
    Label(status_frame,textvariable=status,font="calibre 18 bold",bg="#7ae9fa").pack(side=LEFT, ipadx=10,ipady=10)
    root.mainloop()