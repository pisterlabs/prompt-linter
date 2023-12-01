import openai
import os
import sys

######################
# API KEY GOES HERE
######################

openai.api_key = os.environ['OPENAI_API_KEY']

######################
# API KEY GOES ABOVE
######################

# Credit: https://robsware.github.io/2020/12/27/gpt3 - shoutout to Rob for this awesome context idea
# Too many personality pre-prompts can hinder the performance of this application. It's recommended to keep the initial dialogue between 5 and 6 lines
personality = "MABP is a helpful AI assistant and master malware analyst.\n" \
	"User:Who are you?\n" \
	"MABP:I am MABP, an AI assistant with expert knowledge in malware analysis.\n" \
	"User:What is the first entry in the Image Optional Header?\n" \
	"MABP:The first entry in the Image Optional Header is the magic number determining whether an image is a PE32 or PE32+ executable.\n" #\

	#"User:What is the implication if an AddressOfEntryPoint in one program for a given DLL has a different offset for the same AddressOfEntryPoint in another program?\n" \
	#"MABP:This could be evidence of module stomping from a malicious executable on the system.\n" \
	#"User:What is this shellcode? 0x6F,0x72,0x69,0x67,0x69,0x6E,0x61,0x6C,0x20,0x73,0x68,0x65,0x6C,0x6C,0x63,0x6F,0x64,0x65\n" \
	#"MABP:This shellcode is the hexadecimal representation of an ASCII string that reads as 'original shellcode'.\n"


def ask_gpt3(prompt):
	# Set the model
	model = "text-davinci-003"
	
	modified_prompt = f'{personality}User: {prompt}\nResponse:'
	
	# Make the request to the text-davinci-003 API
	response = openai.Completion.create(engine=model, stop=['Response:'], \
		prompt=modified_prompt, max_tokens=2048, temperature=0.5, \
		top_p=1, frequency_penalty=0, presence_penalty=0.1)

	# Get tokens and cost for the prompt
	tokens = response['usage']['total_tokens']
	print(f"Tokens: {tokens}")
	print(f"Cost: {(tokens / 1000) * 0.02}")

	# Construct the response
	output = response['choices'][0]['text']

	return output # there was a return output[2:] hack here, but I forgot why

def query(prompt):
	try:
		question = prompt
		answer = ask_gpt3(question)
		return answer

	except Exception as e:
		print(repr(e))

if __name__ == "__main__":
	query(prompt)
