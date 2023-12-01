import cohere
co = cohere.Client('L2VMOXwleskZQjVuP5QEe2puJTKNLAzGaRhSEVTK')

# starting message, add question from array
message = "Hello! I'm TutorBo. Time to test you on some questions! Question 1: "

response = co.chat(
    message,
    model="command",
    temperature=0.9
)

answer = response.text

chat_history = []
max_turns = 1000

for _ in range(max_turns):
	# get user input
	message = input("Answer TutorBo's question! ")
	
	# generate a response with the current chat history
	response = co.chat(
		message,
		temperature=0.8,
		chat_history=chat_history
	)
	answer = response.text
		
	print(answer)

	# add message and answer to the chat history
	user_message = {"user_name": "User", "text": message}
	bot_message = {"user_name": "Chatbot", "text": answer}
	
	chat_history.append(user_message)
	chat_history.append(bot_message)
