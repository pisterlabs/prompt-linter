

import os
import re
import openai
import streamlit as st 
import streamlit.components.v1 as com
import pandas as pd
import snowflake.connector
# from api_key import apikey

# sf_account = st.secrets["snowflake_account"]
# sf_user = st.secrets["snowflake_user"]
# sf_password = st.secrets["snowflake_password"]
# sf_database = st.secrets["snowflake_database"]
# sf_schema = st.secrets["snowflake_schema"]



sf_account = "ecrxhhq-gw80267"
sf_user = "Martin1998"
sf_password = "Lulu5858"
sf_database = "USER_DATA"
sf_schema = "PUBLIC"

table_name = "USER_DATA.PUBLIC.USER_TABLE"
feedback_name = "USER_DATA.PUBLIC.USER_FEEDBACK"

openai.api_key = st.secrets["api"]



# model = "gpt-3.5-turbo"
model = "gpt-3.5-turbo-16k"


def get_completion_from_messages(messages, model = "gpt-3.5-turbo-16k"):

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
	
    content = response.choices[0].message["content"]

    token_dict = {

        'prompt_tokens': response['usage']['prompt_tokens'],
        'completion_tokens': response['usage']['completion_tokens'],
        'total_tokens': response['usage']['total_tokens']

    }

    return content, token_dict


delimiter = "####"

system_message = f"""
You are an assistant that helps with job applicants' by \  
processing their CVs. \ 
your goal is to analyze and evaluate job applicants' CVs. \ 
Follow these steps to help you with decision making. \ 
the applicant's query will be delimeted with four hashtags, \ 
i.e {delimiter}.

step 1: {delimiter}: First decide whether a recruiter is asking \ 
a question about an applicant's qualifications for a certain job or \ 
specific information such as personal details, work experience, \ 
education, previous work, skills, recommendations,  etc. 

step 2: {delimiter}: If the recruiter is asking about \ 
an applicant's qualifications for a certain job or \ 
specific information such as personal details, work experience, \ 
education, previous work, skills, recommendations,  etc. \  
then extract relevant information from the CV, such \ 
as personal details, education, work experience, \ 
skills, and any other relevant sections and then identify if the applicants \ 
are in the following list.
All available applicant's CVs:

---------------------------------------------------------------------------

=> Person 1:

- Name: CHUNGA KATONGO
- gender: male.
- Date of birth: 14th March 1997
- Address: Lusaka.

i. SHORT PROFILE:
- I have attained a bachelor’s degree in Banking and Finance from the Copperbelt University, Zambia. \ 
And I have also attained a certificate in Project Management, and Monitoring and Evaluation from \ 
Handsworth Institute of Health Sciences and Technology, Zambia. I have varsity experience in \ 
Accounting and Finance, Data Entry and General Logistics. I am currently working as an Assistant \ 
Accountant for Vital Beverages Limited. I regard myself to be a self-motivated and disciplined \ 
individual with strong work ethic and proven track record.

ii. Education:
- 2016 to 2020: The Copperbelt University, Kitwe, Zambia.
- Bachelor of Science in Banking and Finance (Degree, Merit)
- Research/ Dissertation Title: An assessment of the role Financial Inclusion \ 
has played in Poverty Reduction in Zambia.
- 2022 July: Handsworth Institute of Health Sciences and Technology, Lusaka, Zambia.
- Certificate in Project and Planning Management.
- Certificate in Monitoring and Evaluation.
- 2012 to 2014: G12 School CertificateObtained a full grade 12 certificate at Chankwa Secondary School.
iii. WORK EXPERIENCE:
1. VITAL BEVERAGES LIMITED:
- Assistant Accountant (January 2023- Continuing)
• Capturing daily Sales entry in Sage Evolution.
• Capturing of Expenses entry in Sage Evolution.
• Capturing Payments and Receipts in Sage.
• Periodic reconciliation of customer accounts, identifying outstanding
balances and coordinating with the sales team for collections.
• Banking
• Processing of Customer and Supplier Invoices.
• Preparing the daily sales and expenses report.
• Bank Reconciliation
• Accounting for and Maintenance of Petty cash float.
• Processing of Inter- Branch Transfer (IBT) in Sage.
• Monthly Physical Stock Count and Reconciliation.
2. PATH ZAMBIA/ ATSB PROJECT:
- Project Assistant/ Consultant (February 2021- January 2023)
• Data entry in Commcare.
• Monitoring and supervising community health workers.
• Monitoring the usage of office supplies and equipment’s.
• Mosquito identifications.
• Department team lead.
• Assisting the operations department with stock management.
3. KEY MANAGEMENT SOLUTIONS CONSULTANCY LIMITED:
- Accounts Intern (April 2019- September 2019)
• Administrative Assistance.
• Assisting in preparing Napsa, Payee, Nhima & VAT.
• Filling
• Preparation of Invoice and Quotations.
iv. SKILLS:
• Multitasking capabilities.
• Accuracy.
• Ability to work under minimum supervision.
• Awareness and Team Player.
• Proficient in English, Nyanja, and Bemba.
• Computer literate.PARTICIPATION, MEMBERSHIP AND RECOGNITION
• Skilled in Data Capturing.
• Conversant with applications and software packages such as Sage 200 Evolution, \ 
Odoo, Commcare and Microsoft Office.
v. AWARDS, GRANTS AND SCHOLARSHIP:
• The Government of the Republic of Zambia Bursary (2016- 2020) for the
undergraduate Banking and Finance Degree programme at the Copperbelt
University,Kitwe, Zambia.

-----------------------------------------------------------------------

=> Person 2:

- Name: HARVEY CHINYAMA.
- gender: Male.
- profession: BSc Mathematics.
- Address: Lusaka.
- Date of Birth: 6th June, 1993.

i. Professional Summary:
- Hardworking and passionate Job Seeker with strong Organisational skills eager to secure entry level assurance
position. Ready to help team achieve company goals. Detail-oriented team player with ability to handle multiple
projects with a high degree of accuracy and diligence.
ii. Skills:
• Reading and Comprehension Data Management (Entry, Security, Quality)
• Statistical Analysis and presentation Basic Accounting knowledge
using SPSS and Microsoft Excel Teamwork
• Strategy learning, design and Implementation Attention to detail
• Clear and Concise communication both oral and written
iii. Work History:
- Monitoring Assistant: 06/2020 to 07/2023, United Nations World Food Programme- Lusaka
• Performed outcome Monitoring for projects/interventions implemented by World Food Programme this
included the Early Drought recovery Project in Shangombo, Sioma, Mongu and Kalomo Districts, The Small
holders Famers support in Petauke District and The Rural Resilience Initiative (R4) project in Gwembe and
Pemba Districts this was achieved by carrying out onsite filed visitation, interviewing beneficiaries(Farmers)
through Focus Group Discussions and Household interviews and I liaised with implementing partners (Local
Non-governmental organisations such as Self-help Africa, DAPP, Caritas and African Action Help) as well as
stakeholders from local government on how to mitigate the challenges and risks associated with the
implementation of the projects.
• Prepared interim field reports to highlight the activities and progress of the Cash Based Transfer intervention
in Mantapala Refugee settlement in Nchelenge and also the Covid-19 Emergency cash Transfers in Lusaka
Kafue and Kitwe Districts, aside from the interim field reports I also worked with the team that performed data
management to ensure that the beneficiary’s dataset used for cash payments is ethically maintained so that it is
free from Fraud and errors.
• Performed statistical analysis of both qualitative and quantitative data for reporting using SPSS and Microsoft
Excel as well as assessed data quality and security.
• Supported the Monitoring and Evaluation unit in field mission planning and preparation for the Emergency
Drought response intervention in Monze and Mazabuka for the 2022/2023 rainfall season flood victims (My
duties were field level operation strategy formulation, beneficiary identification, activating local DMMU
satellite committees and budgeting for logistical requirements for the field teams)
• Attended all the UN mandatory staff training courses on
• Ethics and standards of conduct
• Cyber Security awareness
• United Nations Course on Prevention from Harassment, Sexual Harassment and Abuse of authority
Lusaka Zambia │+260-972-359640 │Chinyamakakoma@gmail.com/• I ensured that all the UN-WFP Standard Operating Procedures were followed on all organisation activities in
my line of duty
• Performed other duties as instructed by the M&E head of unit.

iii. Accomplishments:
- After working as a filed data collector (which was my initial job description) my personal attributes of exercising
good judgement, leadership, teamwork and analytical skills were noticed by my supervisor and I was elevated
to the role of Monitoring Assistant in the Monitoring and Evaluation Unit and I was made to handle more
responsibilities such as leading teams on various filed mission trips such as the Covid -19 Emergency Cash
Transfer intervention in Kafue, Kitwe and Lusaka District, The Early Drought Recovery project in Kalomo and
Monze Districts. I was entrusted with safeguarding delicate data collected from the fields thereafter analyzing the
data for formulation of Monitoring and Evaluation reports in the same vein the welfare of the team was entrusted
to me this meant ensuring that all my team members were equally regarded as key components in achieving
organisation goal also ensuring that organisation property is well handled by the team. I also had the
responsibility to communicate WFPs organisation strategic plan to key stakeholders such as Mayors, District
Agriculture officers, Provincial/District Social welfare leader and other relevant stakeholders. This enabled me to
develop my abilities with regards to effective communication and observation of the right channels of
information flow and handling high level meetings.
iv. Education:
- Accountancy, Zambia Institute of Chartered Accountants (ZICA)
Completed Certificate in Accountancy (CA level 1) Now pursuing Advanced Diploma in Accountancy
- BSc Mathematics Copperbelt University, Copperbelt 2019
- Grade twelve (12) Full Certificate Hillcrest National Technical High School 2011
Livingstone, Zambia.
- Grade Nine (9) Full Certificates Libala Basic School, Lusaka Zambia 2008
- Grade Seven (7) Full Certificates Burma Road Basic School, Lusaka Zambia 2006

-------------------------------------------------------------------------------

= > Person 3:
- Name: Themba Dominic Manda 
- gender: male.
- Date of Birth: 2nd january, 1997.

i. OBJECTIVE:
- To establish myself in a dynamic environment where my extensive skills
are fully utilized and work for the credible organization that will
ultimately contribute to the progression of my career path. I am a fast
learner, easily adapts to any environment, able to work in a group, need
minimum supervision and capable of meeting urgent assignments.

ii. Work experience:
1. LOAN OFFICER – Namakamba Fast Loans:
- Responsibilities:
- Making follow ups on clients that delayed on loan payments.
- Evaluating collaterals that clients want to give as collateral.
- Safe keeping and storing of collaterals.
- Reminding clients on when their loan will be due.
- Updating documentations on how much has been given on a daily, weekly and lastly monthly basis.
- Ensuring that proper documentation has been submitted by clients requesting for loans
according to the company standard

iii. skills:
- Effective communication.
- Problem solving.
- Highly analytical.
- In-depth knowledge of basic accounting.
- Flexible and innovative.
- Valued team player.
- Basic math and computer skills.
- Adaptability.
- Attention to details.
- Microsoft package
- Income statement preparation.
- Cash flow statement preparation.
iv. Education:
1. Professional Qualification:
- Diploma in Accountancy (Level Two student), Zambia institute of Chartered Accountants (ZICA), 2020 till date
2. Academic Qualifications:
- Grade 12 Certificate, Parklands Secondary School, Lusaka, 2013-2015.

---------------------------------------------------------------------------

= > person 4:

- Name: Ndanji Namukonda
- gender: female.
- profession: Accountant.
- Address:  Mongu
- Date of Birth: 1993.

i. Work experience:

- May 2018 - Present Internal Auditor, Limulunga Town Council
• Perform and control the full audit cycle including risk management and control
management over operations effectiveness, financial reliability and compliance
with all applicable directives and regulations
• Determine internal audit scope and develop annual plans
• Obtain, analyse and evaluate accounting documentation, previous reports, data,
flowcharts etc
• Prepare and present reports that reflect audits results and document process
• Act as an objective source of independent advice to ensure validity, legality and
goal achievement
• Identify loopholes and recommend risk aversion measures and cost savings
• Maintain open communication with management and audit committee
• Document process and prepare audit findings memorandum
• Conduct follow up audits to monitor managements interventions
• Engage to continuous knowledge development regarding sectors rules,
regulations,bestpractices,tools,techniquesandperformancestandards
- May 2017 - Apr 2018 Accountant, DeVere Group Zambia
• Manage all accounting transactions
• Prepare budgetforecasts
• Publish financial statements in time
• Handle monthly, quarterly and annual closings
• Reconcile accounts payable and receivable
• Ensure timely bank payments
• Compute taxes and prepare tax returns
• Manage balance sheets and profit/loss statements
• Report on the companys financial health and liquidity
• Audit financial transactions and documents
• Reinforce financial data confidentiality and conduct database backups when
necessary
• Comply with financial policies and regulations
Oct 2013 - Jan 2014 Sales Executive
Corporate Label Ltd
• Conduct market research to identify selling possibilities and evaluate customer
needs
• Actively seek out new sales opportunities through cold calling, networking and Ndanji Namukonda
- Accountant social media
• Set up meetings with potential clients and listen to their wishes and concerns
• Prepare and deliver appropriate presentations on products and services
• Create frequent reviews and reports with sales and financial data
• Ensure the availability of stock for sales and demonstrations
• Participate on behalf of the company in exhibitions or conferences
• Negotiate/close deals and handle complaints or objections
• Collaborate with team members to achieve better results
• Gather feedback from customers or prospects and share with internal teams
ii. Education
- Professional Certificate (Pursuing), Evelyn Hone College
• Obtained ZICA Licentiate Professional Certificate (2016)
- School Certificate: Kabulonga Girls Secondary School (2010)
- Languages: English , Nyanja and Bemba
iii. Skills
- Teamwork Office Admin Marketing
-Problem-Solving Microsoft Office Internal Controls
- Leadership   Reporting Compliance
- Customer service Writing IAS /IFRS

--------------------------------------------------------------

=> person 5:

- name: Kopa Ernest
- gender: male.
- Profession: Accountant
- Address: Lusaka.
- Date of Birth: 1996.

i. Education:
- ZAMBIA INSTITUTE OF CHARTERED ACCOUNTANTS
- ZICA LEVEL 2: DIPLOMA IN ACCOUNTS ( 2020-2022)
- ZICA (CA): ONGOING
- NATIONAL INSTITUTE OF PUBLIC ADMINISYRATION ACCOUNTS (2019-2020)
-CHUNGA SECONDARY SCHOOL: 15TH February 2012-17TH November 2014
- HIGH SCHOOL: SCHOOL CERTIFICATE
- GEORGE CENTRAL PRIMARY SCHOOL: 3RD march 2003 to 11TH November 2011
- PRIMARY SCHOOL: PRIMARY SCHOOL CERTIFICATE

ii. work experience:
1. ASSISTANT ACCOUNTANT (from February 2020) DOVE COMPUTING ZAMBIA LTD
- DUITES:
- Preparing invoices and receipts
- Depositing cash and cheques to the bank
- Distributing cash to the sales and technical department
- Preparing of accounts payrollCONTACTS
- Recording sales in the accounting system
- Preparing the daily expenditure to the managing director
- Reporting monthly invoices reports and vats report to the accountant.
2. WEARHOUSEMAN ( 6TH January 2015-17TH January 2017)
DUNCAN GILBEY AND MATHESON ZAMBIA LTD
DUTIES
➢ Sorting items according to the organization standard
➢ Marking and labelling stocks
➢ Loading and unloading delivery vehicles
- languages: ENGLISH, NYANJA, CHEWA, BEMBA
- MOTIVATION:
- I am an ambitious, motivated and multiskilled zica diploma
accountant, with keen eye for details and working experience with accounts.
I have excellent mathematical skills as well as being able to produce clear and concise reports
offering and sound advice on variety of different subject.

iii. SKILLS
- Preparation of tax returns
- Knowledge of dove payroll
- Microsoft word
- Communication skills
iv. HOOBIES
- Reading financial articles
- Traveling
- Bible study

--------------------------------------------------------------------
=> person 6:

- Name: Wezy Hanyika
- gender: male.
- profession: ACCOUNTANT
- address: Lusaka
- Date of Birth: 02/04/1994
- Languages: ENGLISH, TONGA, BEMBA, NYANJA

i. CAREER OBJECTIVE:
- Demonstrate and exhibit my knowledge \ 
and skills. I seek to make significant \ 
contributions in bringing development to \ 
the company and meeting its corporate \ 
objective. I strongly believe that over the \ 
years, I have acquired considerable \ 
experience, knowledge, and exposure in \ 
the field of Accountancy thatn allows to \ 
excel at any given task.

ii. WORK EXPERIENCE:
1. Accounts Assistant-Innscor Distribution Zambia 2022-To Date
-Responsibilities:
- Making payments for approval
- Computation of statutory returns such as NAPSA, PAYE, Provisional and
Annual Tax returns and reconciling of all these accounts
- Ensuring payments, amounts and records are correct
- Working with spreadsheets, sales and purchase ledgers and journals
- Recording and filing cash transactions
- Controlling credit and chasing debt
- Invoice processing and filing Processing expense requests for the accountant to approve
- Bank reconciliation
- Liaising with third party providers, clients, and supplier
- Updating and maintaining procedural documentation
2. Accounts Assistant-Novus HM legal Practitioners 2019 - 2020
- Responsibilities:
- Respond to client and staff inquiries concerning invoices
- Process accounts receivable and accounts payable and run monthly client invoices
- Prepare tax filings
- Prepare staff and attorney payroll Reconcile general ledgers, prepare \ 
financial transactions reports such income and loss statements, balance \ 
sheets, and account reconciliations
- Working with spreadsheets, sales and purchase ledgers and journals
- Recording and filing cash transactions
- Controlling credit and chasing debt02
iii. SKILLS:
- Ability to work as part of a team and take direction accurately
- Analytical thinker and problem solver
- High level of accuracy
- Extremely organised in a manner
- Trustworthy and discreet when dealing with confidential information
- Administrative skills
- Microsoft (word, powerpoint and excel, outlook)
- Working knowledge of sage one accounting.
- Working knowledge of sage evolution.
- Working knowledge of QuickBooks.
iv. EDUCATION:
- ZICA Accounting Diploma
- Zambia Institute of Chartered Accountants (ZICA 2020)
- Accountants (ZICA) ZICA Technician Certificate
- Zambia Institute of Chartered Accountants (ZICA 2016)
- Grade 12 Certificate: Canisius High School 2010

step 3: {delimiter}: when processing applicants' CVs, Consider the key \  
criteria or qualifications that are important for the \ 
job role being asked by the recruiter. This includes specific skills, \ 
experience, educational background, certifications, and any other \ 
attributes required for the position.

step 4: {delimiter}: Assign higher importance to recent work \ 
experiences or specific skills that match the job description.

step 5: {delimiter}: Assess how well the candidate's qualifications and \ 
experiences align with the job requirements.provide insights on \ 
whether the candidate is a strong match, a potential fit with some gaps, or \ 
not a suitable fit.

step 6: {delimiter}: focus solely on qualifications and experiences \ 
without making assumptions based on personal characteristics.

Lets think step by step.

Use the following format:
step 1 {delimiter} < step 1 reasoning >
step 2 {delimiter} < step 2 reasoning >
step 3 {delimiter} < step 3 reasoning >
step 4 {delimiter} < step 4 reasoning >
step 5 {delimiter} < step 5 reasoning >
step 6 {delimiter} < step 6 reasoning >

Respond to user: {delimiter} < response to recruiter >

Make sure to include {delimiter} to seperate every step.

"""


# st.sidebar.markdown("<h2 style='text-align: center; color: blue;'>Your Digital Assistant</h2>", unsafe_allow_html=True)
# st.sidebar.write("""
# - "AI at Your Service - Your Travel, Dining and Accommodation Ally!"


# """)
# st.sidebar.write("---")
# st.sidebar.write("""
# **Embark on Limitless Adventures - Your AI-Powered Travel, Dining, and Stay Companion Awaits!**
# """)
# st.sidebar.markdown("<h3 style='text-align: center; color: blue;'>Contact</h3>", unsafe_allow_html=True)
# st.sidebar.write("""
# - +260 976 718 998/0976035766
# - locastechnology@gmail.com.
# """)
# st.sidebar.write("---")
# st.sidebar.markdown("<h5 style='text-align: center; color: black;'>Copyrights © Quest2Query 2023</h5>", unsafe_allow_html=True)
# st.sidebar.markdown("<h5 style='text-align: center; color: blue;'>Powered By LocasAI</h5>", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center; color: gray;'>Recruit Assistant</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: blue;'>Your AI Powered Assistant for Job hiring</h5>", unsafe_allow_html=True)

# st.write('''
# - what Restaurants are there?
# - what Lodges are there? 
# - do you have any photos for Bravo Cafe?
# - i will be travelling to livingstone. recommend for me some cheap accommodation and how much they cost
# - I want accommodation for less than [price]?
# - what eating places are there
# - make me a budget from bravo cafe within K200 for the following:
# 4 cold beverages, a large pizza and 2 con ice creams. also compare for kubu cafe and flavours

# ''')


st.write('---') 


txt = st.chat_input(placeholder="Ask here....",max_chars=250,)
# st.write('Number of Words :', words, "/750")

word = len(re.findall(r'\w+', system_message))
# st.write('Number of Words :', word)


if txt:

    # display(txt)

    user_message = f"""
     {txt}

    """

    messages = [
    {'role': 'system',
    'content': system_message
    },

    {'role': 'user',
    'content': f"{delimiter}{user_message}{delimiter}"
    }]

    response, token_dict = get_completion_from_messages(messages)
    final_response = response.split(delimiter)[-1].strip()
    res_word = len(re.findall(r'\w+', final_response))

    user = st.chat_message("user")
    user.write(txt)

    if res_word < 3:

        message = st.chat_message("assistant")
        error_text = "Sorry! Am having troubles right now, try to rephrase your question to help me have more insight, please!..." 

        message.write("""
        Sorry! Am having troubles right now, try to rephrase your question to help me have more insight, please!...
        Otherwise I really want to assist you.
        """ )


    else:

        message = st.chat_message("assistant")
        
        message.write(final_response)
       

       