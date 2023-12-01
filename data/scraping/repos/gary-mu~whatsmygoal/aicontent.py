import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

no_repsone = 'Opps sorry, I cannot answer that right now'

def openAIQuery(future_career):
	response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(future_career),
            temperature=0.6,
            max_tokens=200,
        )
	
	if 'choices' in response:
		if len(response['choices']) > 0:
			answer = response['choices'][0]['text']
		else:
			answer = no_repsone
	else:
		answer = no_repsone
		
	return answer



def generate_prompt(future_career):
    return """
    I am a middle school student between age 11-14, what goals can I set if I want to become a {} in the future?
    
    Great question! Here are 5 goals as a middle school student between age 11-14 you can set to become a {} in the future:
    """.format(
        future_career.capitalize(),
        future_career.capitalize(),
    )    