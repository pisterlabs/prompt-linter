import argparse
from typing import Dict, List
from flask import Flask, request, jsonify
import openai
import traceback
import sys
import time
import requests
import re
import copy
import threading

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run API with OpenAI parameters.")
    parser.add_argument("--openai_api_key", required=True, help="OpenAI API key")
    parser.add_argument("--auth_token", default="access_token_to_this_server", help="Authentication token")
    return parser.parse_args()


# Define the Flask app
app = Flask(__name__)

@app.route("/", methods=["POST"])
def chat():
    # Check authentication token
    request_data = request.get_json()
    auth_token = request_data.get("verify_token")
    if auth_token != args.auth_token:
        return jsonify({"error": "Invalid authentication token"}), 401

    # Get messages from the request
    
    messages = request_data.get("messages", [])

    # Call the forward function and get the response
    try:
        response = forward(messages)
    except:
        traceback.print_exc(file=sys.stderr)
        response = "That is an excellent question..."

    # Return the response
    return jsonify({"response": response})




def call_openai(messages, model_name = "gpt-3.5-turbo", temperature = 0.1, max_tokens = 200, top_p = 0.1):

    model_name = model_name or args.model_name
    temperature = temperature or args.temperature
    max_tokens = max_tokens or args.max_tokens
    top_p = top_p or args.top_p

    resp = openai.ChatCompletion.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=1,
                        presence_penalty=1,
                    )["choices"][0]["message"]["content"]

    resp = resp.replace("\n\n","\n")
    return resp


def get_highest_score_response(request_str):
    highest_score = float("-inf")
    highest_score_response = None

    for response in memory[request_str]["all_replies"].keys():
        score = memory[request_str]["all_replies"][response].get("score", -1)
        if score >= highest_score:
            highest_score = score
            highest_score_response = response

    return highest_score_response, memory[request_str]["all_replies"][highest_score_response]

class Get_And_Score():
    def __init__(self):
        self.model_prompts = {

            "openai_1":"Reference back to the question, concise, and demonstrate a clear understanding of the topic.\n\n",
            
            "openai_question":"",

            }
        
    def filter_messages(self, messages):
        filtered = []
        for q in messages:
            if not q["role"] == "system":
                if len(q["content"]) > 200:
                    q["content"] = q["content"][-300:]
                filtered.append(q)
        messages = filtered
        return messages
    
    def forward(self, messages, request_str, model_name, score=False, estimated_score=0.1, filter_messages=True ):
        
        messages_for_generation = copy.deepcopy(messages)
        
        if filter_messages:
            messages_for_generation = self.filter_messages(messages_for_generation)
        
        messages_for_generation[-1]["content"] = self.model_prompts[model_name] + messages_for_generation[-1]["content"]
        

        start_time = time.time()
        
        if "openai" in model_name:
            responses = [call_openai(messages_for_generation, model_name="gpt-3.5-turbo", temperature=0.7, top_p=0.9)]


            
        generation_time = str(time.time()-start_time)[:6]
        for resp in responses:
            with lock:
                memory[request_str]["all_replies"][resp] = {"score":estimated_score,"model":model_name, "generation_time":generation_time}
                all_responses.append(resp)
    
        
        print("Model", model_name, "Generation time", generation_time)
 


def generate_replies(messages, request_string):
        with lock:
            memory[request_string]["times"]["start_gen_time"] =time.time()

        if "Ask one relevant and insightful question about the preceding context." in  messages[-1]["content"] or "Ask a follow up question." in messages[-1]["content"]:
            all_threads = []

            
            all_threads.append(threading.Thread(target=response_generator.forward, args=(messages, request_string, "openai_question",True,0.21)))



            for t in all_threads:
                t.start()

            start_time=time.time()
            for t in all_threads:
                max_wait_time = max(1, 30 - (time.time()-start_time))
                t.join(max_wait_time)
            
            with lock:
                memory[request_string]["status"] = "done"
                memory[request_string]["times"]["done_time"] =time.time()
                
            total_time = time.time() - start_time
            print("\nRequest string:", request_string)
            print("Get and score done", "total time:",total_time)
            print("All generations with score:",memory[request_string]["all_replies"], "\n", flush=True)
        

        else:
            
            
            all_threads = []

            all_threads.append(threading.Thread(target=response_generator.forward, args=(messages, request_string, "openai_1",True,0.5)))

            
            for t in all_threads:
                t.start()

            start_time=time.time()
            for t in all_threads:
                max_wait_time = max(1, 30 - (time.time()-start_time))
                t.join(max_wait_time)
            
            with lock:
                memory[request_string]["status"] = "done"
                memory[request_string]["times"]["done_time"] =time.time()
                
            total_time = time.time() - start_time
            print("\nRequest string:", request_string)
            print("Get and score done", "total time:",total_time)
            print("All generations with score:",memory[request_string]["all_replies"], "\n", flush=True)

def manual_check_erotica(request_str):
    for word in ["cock","sex toy", "large breasts", "pussy", "whore","dildo","dick","penis","bdsm","incest","fetish"," clit","cunt"]:
        if word in request_str.lower():
                print("Auto flag", request_str[:20], flush=True)
                return True
    return False

def forward(messages: List[Dict[str, str]]) -> str:
    global args

    request_str = messages[-1]["content"].replace("\n"," ").replace("  "," ").replace("  "," ").replace("  "," ").replace("  "," ").strip()
    

    if manual_check_erotica(request_str):
        return "I can't respond to your message since it might go against our policy against erotica, violence and propoaganda."
    # print("request_str", request_str[:20], flush=True)
    start_time_processing=time.time()
    
    with lock:
        if request_str in memory:
            in_memory=True
        else:
            in_memory=False
    
    if not in_memory:
        with lock:
            memory[request_str] = {"all_replies": {"That is an excellent question.":{"score" : 0,"model":"default_answer", "generation_time":0}}}
            memory[request_str]["status"] = "generating"
            memory[request_str]["frequency"] = 1
            memory[request_str]["times"] = {"start_time":time.time()}
            
        thread = threading.Thread(target=generate_replies, args=(messages, request_str))
        thread.start()
    else:
        in_memory=True

    timed_out=True
    while time.time()-start_time_processing < 9.0:
        try:
            if memory[request_str]["status"] != "done":
                time.sleep(0.2)
            else:
                timed_out=False
                break
        except:
            traceback.print_exc(file=sys.stderr)
            print("Error row 209")
            break

        

    resp = "That is an excellent question..."
    if request_str in memory:
        if "all_replies" in memory[request_str]:
            resp, resp_dict = get_highest_score_response(request_str)
        if "frequency" in memory[request_str]:
            memory[request_str]["frequency"] = memory[request_str]["frequency"] + 1
    if not in_memory:
        print("\nIn memory:", str(in_memory), "timed_out:", timed_out)
        try:
            print("All replies")
            for key in memory[request_str]["all_replies"].keys():
                print(memory[request_str]["all_replies"][key], key)
            print()
        except:
            print("Answer deleted?")
        print("Current time:", time.time(), "All times", memory[request_str]["times"])
        print("Request string:", request_str[-50:])
        print("Response", resp[:100])
        print("Best model", resp_dict["model"], "BIn memory:", str(in_memory), "timed_out:", timed_out)
        print("Score", resp_dict["score"], "\n", flush=True)

    return resp



if __name__ == "__main__":
    args = parse_arguments()

    memory = {}
    all_responses=[]
    lock = threading.Lock()


    response_generator = Get_And_Score()
    
    
    openai.api_key = args.openai_api_key
    app.run(host="0.0.0.0", port=8008, threaded=True)
