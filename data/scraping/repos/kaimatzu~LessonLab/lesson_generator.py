from dotenv import load_dotenv
import os
# import openai

# new lesson generation
import zmq
import time
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

def generate_lesson(source):
    

    # openai.api_key = os.getenv("API_KEY")

    # prompt = "Can you make a markdown format lesson based on this source: " + source

	# # TODO: https://app.clickup.com/t/86ctyuedm make OpenAI focus on a particular subject right now it's not focusing on a topic given in the "focus topic" specification

    # completion = client.openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
    # messages=[
    #     {"role": "system", "content": "You are a college teacher."},
    #     {"role": "user", "content": prompt}
    #     ]
    # )
    # reply = completion['choices'][0]['message']['content']
    return "something"

# new lesson generation
def generate_lesson_stream(source):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5555")
    
    start_time = time.time()
    prompt = "Can you make a markdown format lesson based on this source: " + source

    response  = client.chat.completions.create(
        stream = True,
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a college teacher."},
            {"role": "user", "content": prompt}
        ]
    )
    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []
    # iterate through the stream of events
    for chunk in response:
        try:
            chunk_time = time.time() - start_time  # calculate the time delay of the chunk
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk.choices[0].delta.content # extract the message
            collected_messages.append(chunk_message)  # save the message
            # print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

            socket.send_string(chunk_message)
            print(f"Sent to Rust: {chunk_message}")

            # Wait for acknowledgment from Rust
            ack = socket.recv_string()
            print(f"Received ACK from Rust: {ack}")
        except Exception as e:
            print(f"Error processing chunk: {e}")
    
    # for i in range(25):
    #     time.sleep(0.25)
    #     socket.send_string("e")
    #     print(f"Sent to Rust: e")

    #     # Wait for acknowledgment from Rust
    #     ack = socket.recv_string()
    #     print(f"Received ACK from Rust: {ack}")
        
    print(f"Finished generation.")
     
    # Send a finish message
    socket.send_string("[LL_END_STREAM]")
    print(f"Sent to Rust: Exit message")
    
    # Wait for acknowledgment from Rust
    ack = socket.recv_string()
    print(f"Received exit ACK from Rust: {ack}")
    
    socket.close()
    context.term()
    
def testrun():
    print("Fucking python")
    
testrun()