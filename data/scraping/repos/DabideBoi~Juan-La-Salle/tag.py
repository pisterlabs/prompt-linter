import openai
OPENAI_API_KEY = 'key here please'
openai.api_key = OPENAI_API_KEY

#prompt = 
"""
Instruction: Categorize the text to one available value. Only choose from the values given. Do not answer with any explanation or note. The available values are:
greeting
Your Name
General Kenobi
noanswer 
PreReg
reserve
regular student
shift
where grades?
goodbye
thanks
pre-registration
reservation
regular student
shift
gone
access registration
program code and section code
class standing
transfer
officially enrolled
transaction 2
online enrollment limit
cannot add subject
program code and section
name update
graduation application
graduation payments and fee
pay w/o attendance
graduation academic honors
entry cards/invitations graduation rites
graduate first semester or summer
commence ceremony schedule
Transcript of Records
electronic Transcript of Records
unofficial Transcript of Records
requests Transcript of Records
schedule Transcript of Records
hold Transcript of Records
diploma
lost diploma
Parent's
data Registrar
change grade
failed deadline
students not in class
Where to pay
When to pay?
Story

Text:
"""

prompt = """
Instruction: Categorize the text to one available value. Only choose from the values given. Do not answer with any explanation or note. The available values are:
    "greeting": Greeting
    "your_name": Asking bot's name
    "noanswer": No answer
    "pre_registration": Pre-registration
    "reservation": Reservation
    "regular_student": Regular student
    "shift": Shift
    "grades": Grade inquiry
    "goodbye": Goodbye
    "thanks": Thanks
    "program_section_code": Program and section code
    "class_standing": Class standing
    "transfer": Transfer inquiries
    "enrollment_status": Enrollment status
    "enrollment_limit": Enrollment limit
    "subject_add_error": Cannot add subject
    "subject_details": Subject details
    "personal_info_update": Personal information update
    "graduation": Graduation inquiries
    "payment_concerns": Payment concerns
    "graduation_ceremony": Graduation ceremony
    "transcript_requests": Transcript requests
    "diploma_concerns": Diploma concerns
    "parents_concerns": Parents' concerns
    "registrar_concerns": Registrar's concerns
    "grade_change": Grade change
    "deadline_concerns": Deadline concerns
    "attendance_concerns": Attendance concerns
    "payment_info": Payment information

Text:
"""

def CLASI(msg):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    max_tokens=10,
    temperature=1.2,
    messages=[
            {"role": "system", "content": prompt + msg},
        ])
    print(msg)
    reply = response['choices'][0]['message']['content']
    print(reply)
    return reply

CLASI("Where to pay for graduation?")
