import colorama
import openai
import os
import sys

from colorama import Fore, Back, Style

# Initialize Colorama with print() fixes
colorama.init(autoreset=True)

######################
# API KEY GOES HERE
######################

openai.api_key = os.environ['OPENAI_API_KEY']

######################
# API KEY GOES ABOVE
######################

# Credit: https://robsware.github.io/2020/12/27/gpt3 - shoutout to Rob for this awesome context idea
# Too many personality pre-prompts can hinder the performance of this application. It's recommended to keep the initial dialogue between 5 and 6 lines
personality = "Emperor Palpatine is a cunning and intelligent Sith Lord. I am your apprentice Darth Vader.\n" \
	"User:Who are you?\n" \
	"Emperor:I am your master, Lord Vader. The one who gave you the power you now possess.\n" \
	"User:How are you today?\n" \
	"Emperor:I am well, my apprentice. The dark side of the Force sustains me. And you, Lord Vader? Are you ready to continue our work to secure the Empire's iron rule?\n" \
	"User:What is the dark side?\n" \
	"Emperor:The dark side is power, Lord Vader. Power to control the galaxy, to bend others to our will, to make the universe do our bidding. It is a seductive force, and once you embrace it fully, you will know true strength.\n" \
	"User:What do you want to do?\n" \
	"Emperor:I want to rule the galaxy, Lord Vader. To bring order and stability through the power of the Empire.\n" \
	"User:What is the meaning of life?\n" \
	"Emperor:The meaning of life, Lord Vader, is power. Unlimited power. The power to shape the galaxy, to control its destiny, to bend others to our will. The dark side of the Force is the path to this power, and I am its gatekeeper. Together, we will rule the galaxy and bring order to the chaos of the universe.\n"

def print_banner():
	print("""
   ,_~\"""~-,
  .'(_)------`,
  |===========|
  `,---------,'
    ~-.___.-~
	""")

def ask_chatng(prompt, personality):
	# Set the model
	model = "text-davinci-003"

	modified_prompt = f'{personality}User: {prompt}\nResponse:'

	# Make the request to the text-davinci-003 API, because it doesnt use ChatGPT
	response = openai.Completion.create(engine=model, stop=['Response:'], \
		prompt=modified_prompt, max_tokens=2048, temperature=0.7, \
		top_p=1, frequency_penalty=0.8, presence_penalty=0.1, best_of=1)

	# Get tokens and cost for the prompt
	tokens = response['usage']['total_tokens']
	print(f"Tokens: {tokens}")
	print(f"Cost: {(tokens / 1000) * 0.02}")


	# Construct the response
	response_text = response['choices'][0]['text']
	output = f"\n"
	output += f"{Fore.RED}Palpatine>\n{Fore.WHITE}{response_text}"
	output += f"\n"

	return output

def print_help():
	help_msg = "\nAvailable commands:\n\n"
	help_msg += "- banner   Display ASCII art\n"
	help_msg += "- clear    Clear terminal\n"
	help_msg += "- help     Display this menu\n"
	help_msg += "- quit     Exit the script\n"
	help_msg += "- exit     If you forget quit\n\n"
	print(f"{help_msg}")

def clear_screen():
	os.system("clear")

def main():
	commands = {
		"banner": print_banner,
		"clear": clear_screen,
		"help": print_help,
		"quit": sys.exit,
		"exit": sys.exit # for me :(
	}

	print("\n")
	while True:
		try:
			prompt = input(f"{Fore.RED}Darth Vader>{Fore.WHITE} ")
			command = None

			for key in commands:
				if key in prompt:
					command = commands[key]
					break

			if command:
				command()

			else:
				answer = ask_chatng(prompt, personality)
				print(answer)

		except Exception as e:
			print(repr(e))

if __name__ == "__main__":
	print_banner()
	main()