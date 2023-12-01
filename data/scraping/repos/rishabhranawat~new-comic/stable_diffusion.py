import requests
import os
import openai

# get response from the following post request:
# curl -X POST https://api.runpod.ai/v2/stable-diffusion-v1/run \
# -H 'Content-Type: application/json'                             \
# -H 'Authorization: Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'    \
# -d '{"input": {"prompt": "a cute magical flying dog, fantasy art drawn by disney concept artists"}}'

runpod_api_key = os.getenv("RUNPOD_API_KEY", "")
if runpod_api_key == "":
    raise Exception("runpod api key not set.")

stable_diffusion_url = "https://api.runpod.ai/v2/stable-diffusion-v1/run"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + runpod_api_key
}


def get_job_id(prompt: str):

    data = {
        "input": {
            "prompt": prompt
        }
    }        

    response = requests.post(stable_diffusion_url, headers=headers, json=data)
    job_id = response.json()["id"]
    return job_id

def get_response(job_id: str):
    get_request_url = 'https://api.runpod.ai/v1/stable-diffusion-v1/status/' + job_id
    response = requests.get(get_request_url, headers=headers)
    return response.json()

if __name__ == '__main__':
    job_id = get_job_id("a cute magical flying dog, fantasy art drawn by disney concept artists")
    while True:
        input('press enter to run get_response')
        print(get_response(job_id))





