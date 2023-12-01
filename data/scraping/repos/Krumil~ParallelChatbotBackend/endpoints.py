from threading import Thread
from queue import Queue, Empty
from flask import jsonify, request, stream_with_context, Response
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from utilities import initialize_bot
from dotenv import load_dotenv
import os
import json
import time


load_dotenv()

users = {}  # Dictionary to store user queues based on user_id
agents = {}  # Dictionary to store agents based on user_id
openai_api_key = os.environ["OPENAI_API_KEY"]


# Queue Callback
class QueueCallback(BaseCallbackHandler):
	def __init__(self, queue):
		self.queue = queue
		self.done_count = 0  # Added this line
		
	def on_chat_model_start(self, *args, **kwargs) -> None:
		print("Chat model started")

	def on_llm_new_token(self, token: str, **kwargs) -> None:
		self.queue.put(token)

	def on_llm_end(self, *args, **kwargs) -> None:
		self.done_count += 1  # Increment the done_count
		if self.done_count == 2:  # Check if it's the second time
			self.queue.put('[DONE]')

# Stream Function
def stream(agent, user_input, user_queue):
	def task():
		agent({"input": user_input})

	thread = Thread(target=task)
	thread.start()

	while True:
		try:
			next_token = user_queue.get(True, timeout=2)  # Added timeout here
			if next_token == '[DONE]':  # Check for end signal
				yield f"data: {json.dumps(next_token)}\n\n"
				break
			if next_token:
				yield f"data: {json.dumps(next_token)}\n\n"
		except Empty:
			time.sleep(0.1)  # Sleep for a short time before trying again
			continue


def query_bot_endpoint():
	user_id = request.args.get('user_id')
	if not user_id:
		return jsonify({"error": "User ID is required"}), 400

	user_input = request.args.get('prompt', '')
	if not user_input:
		return jsonify({"error": "User input is required"}), 400
	
	user_queue = users.setdefault(user_id, Queue())

	# Initialize LLM based on user status
	llm = None
	if user_id not in agents:
		llm = ChatOpenAI(
			streaming=True, 
			callbacks=[QueueCallback(user_queue)], 
			model="gpt-3.5-turbo-16k"
		)
		current_time = time.time()
		agent = initialize_bot(llm)
		agents[user_id] = {
			'agent': agent,
			'timestamp': current_time
		}
	else:
		agent = agents[user_id]['agent']

	return Response(stream_with_context(stream(agent, user_input, user_queue)), content_type='text/event-stream')



def cleanup_old_agents():
	while True:
		current_time = time.time()
		agents_to_remove = []

		for user_id, agent_data in agents.items():
			agent_creation_time = agent_data['timestamp']
			if current_time - agent_creation_time > 86400: 
				agents_to_remove.append(user_id)

		for user_id in agents_to_remove:
			del agents[user_id]

		time.sleep(60)

