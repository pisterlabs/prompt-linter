import openai 
# from rs_pdf import extracted_text


system_message = f"""
Your task is to summarise given resume into sub fields like Education, Projects,Achievements, Skills\
dont answer in lengthy text just ton the point bullet points\
SUm up experience in few lines \
Respond in json format for subfields of Job Description\
Give output in dictionary with sub fields as keys\
"""

# data = extracted_text

data = " i am demo data"

def create_message_rs(data):
    messages =  [  
    {'role':'system',
    'content': """
        Your task is to summarise given resume into sub fields like Education, Projects,Achievements, Skills\
        dont answer in lengthy text just ton the point bullet points\
        SUm up experience in few lines \
        Respond in json format for subfields of Job Description\
        Give output in dictionary with sub fields as keys\
        """},   
    {'role':'assistant',
    'content': f"""Relevant resume content: \n
    {data}"""},   
    ]
    return messages



messages =  [  
{'role':'system',
 'content': system_message},   
{'role':'assistant',
 'content': f"""Relevant resume content: \n
  {data}"""},   
]

#  changed name to rs_final
def rs_final(messages, model="gpt-3.5-turbo", temperature=0.5, max_tokens=2000):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=500, 
    )
    return response.choices[0].message["content"]

# print()
#  comment out current examples
# summ_res = get_completion_from_messages(messages)
# print(summ_res)



# { "Education": { "Degree": "BS in Management Information Systems", "University": "New York University", "GPA": "3.922", "Honors": "Magna Cum Laude, Alpha Sigma Lambda Honor Society, Dean's List: Fall 2001, Spring 2002", "Coursework": [ "Statistical Methods", "Economics", "Database Design", "System Analysis and Design", "Business Organization and Management", "Management Information Systems", "Object Oriented Analysis", "Interactive Design", "IT Networking" ] }, "Projects": [ "Initiated and managed program to test incoming radiation detection equipment", "Managed data collection and test efforts for a roadway deployed radiation detector prototype", "Appointed Data Collection Lead for Advanced Spectroscopic Portal test campaign", "Managed technology installation, integration, and data analysis for radiation detection data" ], "Achievements": [ "Initiation and design of equipment testing program leading to Memorandum of Understanding", "Recipient of numerous awards for both team and individual performance", "Completed government training and certification program for Test and Evaluation Manager Level II", "Developed working relationships with many State and local responder agencies" ], "Skills": [ "Project Management", "Information Design", "Relationship Building", "Information Technology", "Procedure Development", "Scheduling", "Website Design", "MS Office Suite", "Training Development", "Quality Processes", "Event Planning", "Work Breakdown Structures" ], "Experience": "More than eight years of progressive management experience and repeated success in developing project initiatives, directing project plans, and achieving performance targets." }
# { "Job brief": "We are looking for a qualified IT Technician that will install and maintain computer systems and networks aiming for the highest functionality. You will also “train” users of the systems to make appropriate and safe usage of the IT infrastructure.", "Responsibilities": [ "Set up workstations with computers and necessary peripheral devices (routers, printers etc.)", "Check computer hardware (HDD, mouses, keyboards etc.) to ensure functionality", "Install and configure appropriate software and functions according to specifications", "Develop and maintain local networks in ways that optimize performance", "Ensure security and privacy of networks and computer systems", "Provide orientation and guidance to users on how to operate new software and computer equipment", "Organize and schedule upgrades and maintenance without deterring others from completing their work", "Perform troubleshooting to diagnose and resolve problems (repair or replace parts, debugging etc.)", "Maintain records/logs of repairs and fixes and maintenance schedule", "Identify computer or network equipment shortages and place orders" ], "Skills and Requirements": [ "Proven experience as IT Technician or relevant position", "Excellent diagnostic and problem solving skills", "Excellent communication ability", "Outstanding organizational and time-management skills", "In depth understanding of diverse computer systems and networks", "Good knowledge of internet security and data privacy principles", "Degree in Computer Science, engineering or relevant field", "Certification as IT Technician will be an advantage (e.g. CompTIA A+, Microsoft Certified IT Professional)" ] }
