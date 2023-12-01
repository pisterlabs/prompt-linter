# libraryes
import os
import openai
import streamlit as st

from dotenv import load_dotenv #to resd the key
load_dotenv()
openai.api_key=st.secrets["auth_key"]






#functions


Template="""



Side Effects:

-Nausea - feeling sick or queasy.

-Headache - a continuous pain in the head.

-Diarrhea - frequent, loose, or watery bowel movements.

-Dizziness - a sensation of spinning or lightheadedness.

-Constipation - difficulty in passing stools or infrequent bowel movements.

-Increased blood pressure - a higher than normal blood pressure reading.

-Urinary tract infection - an infection affecting the urinary system.

-Upper respiratory tract infection - an infection affecting the nose, throat, sinuses, or lungs.

Uses:

-Mirzagen is primarily used for the treatment of overactive bladder (OAB) symptoms, such as frequent urination, urgent need to urinate, and sudden urine leakage.

-It helps to relax the muscles in the bladder, allowing for increased bladder capacity and decreased frequency of urination.

-Mirzagen may also be used off-label for the treatment of other conditions as determined by a healthcare professional.

"""


#Get Qustion function
def GetQustion(name):
    prompt=f"""
    You will be given a name of a medicine delimted by four backquotes.
    the response must containe :all Sid affects,and all uses.
    Display Side affects,and uses in diffrent pargraph.
    if you recived name that is not related to a medicine display:"That is not a name of medicine".
    
    this is an example of the response: {Template}
    


    The name: ````{name}````
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content":prompt,

            },
        ],
    )
    return response["choices"][0]["message"]["content"]


# main function
def main():
    
    
    # with streamlit
    
    st.set_page_config(
        page_title="My Medicine",
        page_icon="💊"
        
    )
    
    
    # st.image("C:\Users\ahmed\Desktop\CS\vscode\medicine.png" , width=500)
    
    st.header("My Medicine App")
    st.write("This app uses OpenAI's GPT-3 to help you to know what are the side effects and uses of your medicine")
    st.divider()
    
    medicine=st.text_input(label="Please enter the Generic Name of your medicine: ",placeholder=None)
    button=st.button(label="Submite")
    if button:
        st.markdown(GetQustion(medicine))



########################################################################################






# execute code
if __name__== "__main__":
    main()
