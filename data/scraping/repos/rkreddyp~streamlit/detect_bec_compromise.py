#from appsmills.streamlit_apps 
from helpers import openai_helpers
import streamlit as st
import numpy as np
from random import randrange
import openai,boto3,urllib, requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import re,json
## 

at = """ these are the attack techniques for business email compromise - - Exploiting Trusted Relationships
    - To urge victims to take quick action on email requests, attackers make a concerted effort to exploit an existing trusted relationship. Exploitation can take many forms, such as a vendor requesting invoice payments, an executive requesting iTunes gift cards, or an [employee sharing new payroll direct deposit details.
- Replicating Common Workflows
    - An organization and its employees execute an endless number of business workflows each day, many of which rely on automation, and many of which are conducted over email. The more times employees are exposed to these workflows, the quicker they execute tasks from muscle memory. BEC attacks [try to replicate these day-to-day workflows]to get victims to act before they think.
- Compromised workflows include:
    - Emails requesting a password reset
    - Emails pretending to share files and spreadsheets
    - Emails from commonly used apps asking users to grant them access
- Suspicious Attachments
    - Suspicious attachments in email attacks are often associated with malware. However, attachments used in BEC attacks forego malware in exchange for fake invoices and other social engineering tactics that add to the conversation’s legitimacy. These attachments are lures designed to ensnare targets further.
- Socially Engineered Content and Subject Lines
    - BEC emails often rely on subject lines that convey urgency or familiarity and aim to induce quick action.
    - Common terms used in subject lines include:
        - Request
        - Overdue
        - Hello FirstName
        - Payments
        - Immediate Action
    - Email content often follows along the same vein of trickery, with manipulative language that pulls strings to make specific, seemingly innocent requests. Instead of using phishing links, BEC attackers use language as the payload.
- Leveraging Free Software
    - Attackers make use of freely available software to lend BEC scams an air of legitimacy and help emails sneak past security technologies that block known bad links and domains.
    - For example, attackers use SendGrid to create spoofed email addresses and Google Sites to stand up phishing pages.
   
"""

bc = """ these are the categories of business email compromise - - CEO Fraud
    - Attackers impersonate the CEO or executive of a company. As the CEO, they request that an employee within the accounting or finance department transfer funds to an attacker-controlled account.
- Lawyer Impersonation
    - Attackers pose as a lawyer or legal representative, often over the phone or email. These attacks’ common targets are lower-level employees who may not have the knowledge or experience to question the validity of an urgent legal request.
- Data Theft
    - Data theft attacks typically target HR personnel to obtain personal information about a company’s CEO or other high-ranking executives. The attackers can then use the data in future attacks like CEO fraud.
- Email Account Compromise
    - In an [email account compromise]attack, an employee’s email account is hacked and used to request payments from vendors. The money is then sent to attacker-controlled bank accounts.
- Vendor Email Compromise
    - Companies with foreign suppliers are common targets of [vendor email compromise] Attackers pose as suppliers, request payment for a fake invoice, then transfer the money to a fraudulent account.

"""

rank = """
urgency, lack of detail, attachments, generic salutation, unusual requests, spelling and grammar.

"""

et = """ Hello John, \n \n We detected something unusual to use an application to sign in to your Windows Computer. We have found suspicious login attempt on your windows computer through an unknown source. When our security officers investigated, it was found out that someone from foreign I.P Address was trying to make a prohibited connection on your network which can corrupt your windows license key.

If you’re not sure this was you, a malicious user might trying to access your network. Please review your recent activity and we'll help you take corrective action. Please contact Security Communication Center and report to us immediately.1-800-816 0380 or substitute you can also visit the Website: https://www.microsoft.com/ and fill out the consumer complaint form. Once you call, please provide your Reference no: AZ 1190 in order for technicians to assist you better.
Our Microsoft certified technician will provide you the best resolution. You have received this mandatory email service announcement to update you about important changes to your Windows Device. \n \n Thanks, Support Team. 

"""


def draw_chart(s_prompt):
    res = openai_helpers.response(s_prompt)
    jsonres = json.loads(res.split('Verdict:')[0])  
    cols = [ "urgency", "lack of detail", "attachments", "generic salutation",   "unusual requests", "spelling and grammar"  ]
    df = pd.DataFrame( list(jsonres.items()) , columns=['Phishing Characterstic', 'Probability'])
    #st.dataframe(df)
    st.subheader ('Phishing Analysis Summary')
    pdf = df [ df['Phishing Characterstic'].str.contains ("verdict|phishing category|attack technique category") == False ]
    #fig = px.bar(pdf.sort_values(by='Probability'), x='Phishing Characterstic', y='Probability', color='Probability', color_continuous_scale=px.colors.sequential.Oryel,
                 #labels={'Probability':'Probability of Phishing'}, height=400)
    #fig.update_layout(title={
        #'text': "Phishing Analysis",
        #'font': {'size':18}
    #})
    fig = px.scatter_polar(pdf, r="Probability", theta="Phishing Characterstic", color='Probability', color_discrete_sequence=px.colors.sequential.Plasma_r,template="plotly_dark")

    st.write ("Verdict:" + str (df.loc [df ['Phishing Characterstic'].str.contains('verdict')]['Probability'].tolist()) )
    st.write ("Phishing category:" + str (df.loc [df ['Phishing Characterstic'] == 'phishing category']['Probability'].tolist()) )
    st.write ("Attack technique category:" + str(df.loc [df ['Phishing Characterstic'] == 'attack technique category']['Probability'].tolist()))
    #st.write ("Phishing category:" + df['phishing category'].tolist()[0] )
    #st.write ("attack technique category:"+ df['attack technique category'].tolist()[0] )
    st.plotly_chart(fig)



def display_text () :

    button_name = "Check email for me !! "
    response_while = "Right on it, it should be around 2-5 seconds ..."
    response_after = "Here you go ...  "


    email_txt = st.text_area("Paste the email you want to check in place of the sample email below and click the check email button (or click the button with the sample email to check out" , value=et, height=400)
    tab_button=st.button(button_name , key = "1")
    if tab_button:
        
        #r = openai_helpers.response( bc )
        #r = openai_helpers.response( at )

        #prompt = " .determine if the below email is a business email compromise,  tell me the reasons and give me a bullet list of ranks (rank as high, medium, low) it in these categories: " + rank + ", categorize it and tell me the attack technique as well - "

        #email_txt = prompt + email_txt
        #penai_helpers.get_write_response ( bc + "." + at + "." + email_txt)
   
        prompt = "you are an email threat detection engine.  determine if the below email is phishing based on urgency, lack of detail, attachments, generic salutation, unusual requests, spelling and grammar , give the output in just one json string (do not include any data after the json) with urgency, lack of detail, attachments, generic salutation, unusual requests, spelling and grammar  as numerical probability key value pairs and verdict, phishing category and attack technique category as string values:"
        
        s_prompt = prompt + email_txt
        response_while = "Right on it, it should be around 2-5 seconds ..."
        response_after = "Here you go ...  "

        
        with st.spinner ( response_while ) :
            draw_chart(s_prompt)
    
        st.subheader ('Full Explanation')
        prompt = " .determine if the below email is a business email compromise,  tell me the reasons , categorize it and tell me the attack technique as well, do not give json as output - "
        prompt = ". determine if the below email is phishing based on urgency, lack of detail, attachments, generic salutation, unusual requests, spelling and grammar, give detailed reasons. "
        s_prompt = prompt + email_txt
        openai_helpers.get_write_response ( bc + "." + at + "." + s_prompt)
   

st.subheader ('Check Emails for BEC Attacks')
           
display_text()

    
