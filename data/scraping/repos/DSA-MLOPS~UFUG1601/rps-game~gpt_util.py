
import openai
import os

# Set openai.api_key to the OPENAI environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]
SYSTEM_MSG = """You are a TA for a first year students. 
You are nice and knowledgable. Provide detailed feedback to the student.
"""

def gpt_grade(hw_desc, student_code, stdout, stderr):

    content = f"""Grade this howework as a TA.
Provide the homewor score [x out of 10] in the first line.
Then, provide detailed reasons for the score.
Please DO NOT PROVIDE solutions. NEVER!
Use markdown syntax to format your feedback.

### Homework Description:
{hw_desc}

### Student Code:
{student_code}

### Output:
{stdout}

### Error:
{stderr}
"""
    
    msgs = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role":"user", "content": content}]
  
    response= openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=msgs)
    status_code = response["choices"][0]["finish_reason"]
    assert status_code == "stop", f"The status code was {status_code}."
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    hw_desc = """
Write code to print out the followings:
*
**
***
****
***
**
*
"""

    student_code = """  
for i in range(1, 5):
    print("*" * i)
for i in range(5, 0, -1):
    print("*" * i)
"""
    stdout = """            
*   
**

***
****

***
**

*
"""
    stderr = ""
    print(gpt_grade(hw_desc, student_code, stdout, stderr))

        
