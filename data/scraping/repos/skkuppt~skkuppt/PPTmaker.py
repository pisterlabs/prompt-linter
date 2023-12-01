from openai import OpenAI
import os

# prompt

def make_prompt(topic:str, details:str)->str:
    prompt = f"""
    make a powerpoint presentation about {topic} with the following details:
    {details}
    In the structure of:
    Slide 1:
    Title:
    Content:
    Slide 2:
    Title:
    Content:
    ...  
    Slide n:
    Title:
    Content:
    Only write the title of the slides and its content. 
    """
    return prompt

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


def gpt_pptmaker(topic, details):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            #model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are making ppt for the class, skilled in making index and its belongings.\
                showing only the title and content of the slides without other flowery words or summary or etc."},
                {"role": "user", "content": make_prompt(topic, details)}
            ],
            temperature=0,
            max_tokens=1024,
        )
    	# get format of messege contetn
    	# print(type(completion.choices[0].message.content))
    	# get the response content only
    	#return completion.choices[0].message.content

		#  Handle Response
        # completioin method는 indexing이 불가능합니다.
        if completion.choices[0]:
            return completion.choices[0].message.content
        else:
            return "No content received from the API."
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    topic = "What is the meaning of life?"
    details = "The meaning of life is to be happy and useful."
    apikey = os.environ.get("OPENAI_API_KEY")
    print(gpt_pptmaker(topic, details))
