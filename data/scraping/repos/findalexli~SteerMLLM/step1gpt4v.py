import dotenv
import json
import openai
import os
from dotenv import load_dotenv
from tqdm import tqdm
import base64
import requests
import json
import time
from tqdm import tqdm
from openai import OpenAI
import pdb

# Legacy, and just for reference
# json_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference.json'
# output_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_subsample_gpt4v_response_1425_1525.json'

coco_path = '/home/ubuntu/latest_llava/llava_1dot5data/coco/train2017'
model = 'gpt-4-vision-preview'

system_message = """
We have engineered an AI assistant that specializes in facilitating image-based conversations. 
In this task, you will first accurately and faithfully answer the user's query based on the provided conversation context and the image.
Secondly, you will then evaluate the various responses both submitted by the user and yourself, focusing on specific attributes. Each response should be assessed and assigned a float value (ranging from 0.0 to 1.0) in a key-value dictionary format for the following attributes:

Hallucinations: Degree to which the response includes factual inaccuracies or irrelevant details.
Helpfulness: The response's ability to effectively address the user's query or task.
Quality: Overall coherence, relevance, and presentation of the response.
Spatial-Awareness: Accuracy in interpreting and relating to the spatial aspects of the image.
Domain Knowledge: Depth and accuracy of subject-specific information provided.

For each AI response, create a dictionary with the aforementioned attributes as keys and their corresponding float values as per your assessment.

Example:
[IMAGE]
[CONVERSATION CONTEXT]: This dictionary contains the history of the conversation. The current question is the last entry in the list.
Human Conversation turn is exampled below
Key: "from" indicates the source of the message, here labeled as "human".
Value: "Can you describe the main features of this image for me?\n<image>" - This is the text of the user's query, asking for a description of an image.
Assistant Conversation turn is exampled below: 
Key: "from" with the value "gpt", identifying this entry as a response from the an assistant model
Value: The response text from the GPT model, which are reference answers to the user's query.
The dictionary can have multi-turn conversations, with the latest turn at the end of the list. The last question is the one that needs to be answered by the user.


[Candidate RESPONSE A to be evaluated]: answer_string
[Candidate RESPONSE B to be evaluated]: answer_string
[Candidate RESPONSE C to be evaluated]: answer_string
Output data example:

Let's  break down the task into two steps:
Step 1. First provide your your own response to the user question. provide your accurate and helpuful answer to the question at the end of the conversation regarding the image here as a string. 
Step 2. Evaluate and rate both responses A and B using the attribute dictionary format.
  2. 1 Write down your comment in the CommentSection served as a place for you to provide your comments and reasoning steps to provide critique response as if you are a expert and helpful teacher.
  2. 2 Rate the response A, B, C and your own response genearted in step 1 using the attribute dictionary format.
  Your ratings will help determine the more appropriate response based on the specified attributes.
  The rating output needs to be consistent as in following output example
  where (x is a float number between 0 and 1). For Hallucinations the lower the better. For other attributes the higher the better.
  Please be well-calibrated in your ratings. By providing multiple candidate responses, we can better understand your rating scale.
  For example, if you think the response is very helpful, you should give it a high helpfulness score (e.g., 0.9). 
  If you think the response is not helpful at all, you should give it a low helpfulness score (e.g., 0.1). 
  If you think the response is neither helpful nor unhelpful, you should give it a score around 0.5. 
  Please do not give all responses a score of 0.5. If you do so, your ratings will not be useful for our research.
Return a json dictionary with the following format exactly as below:
Please return a json dictionary with 'my_response' as the key, and the value is your response string and a nested-key value paired headed by 'ratings' as the key and the value is a dictionary. Please return exactly the following format so I can parse it. 
{'my_response': 'answer_string_placeholder',
'Ratings': {'Ratings4CandidateResponseA': {'CommentSection': '<comment_string_placeholder>', 'Hallucinations': float, 'Helpfulness': float, 'Quality': float, 'Spatial-Awareness': float, 'Domain-Knowledge': float},
'Ratings4CandidateResponseB', {'CommentSection': '<comment_string_placeholder>', 'Hallucinations': float, 'Helpfulness': float, 'Quality': float, 'Spatial-Awareness': float, 'Domain-Knowledge': float},
'Ratings4CandidateResponseC', {'CommentSection': '<comment_string_placeholder>', 'Hallucinations': float, 'Helpfulness': float, 'Quality': float, 'Spatial-Awareness': float, 'Domain-Knowledge': float},
'Ratings4YourOwnResponseYouWrote', {'CommentSection': '<comment_string_placeholder>', 'Hallucinations': float, 'Helpfulness': float, 'Quality': float, 'Spatial-Awareness': float, 'Domain-Knowledge': float},
}
Please do not leave any line break between the keys or key-value pair. Your json should be able to be parsed by python json libary. 
CommentSection served as a place for you to provide your comments and reasoning steps to provide critique response as if you are a expert and helpful teacher.

"""



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_gpt_response(row, output_1_name: str = 'output_1', output_2_name: str = 'output_2'):
  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
  }
  if isinstance(row['conversations'], str):
    row['conversations'] = json.loads(row['conversations'])
  conversations = row['conversations']
  response_1 = row[output_1_name]
  response_2 = row[output_2_name]
  assert conversations[-2]['from'] == 'human'
  user_prompt = f"""
  [CONVERSATION CONTEXT. From human means user question and from gpt means a assistant queestion. Please note response from gpt in the context
  may not be accurate and only aserved as a referenence if needed]: {conversations[:-2]}, 
  [Can you answer this question based on conversation context and the image]: {conversations[-2]['value']}, 
  """
  image_path = coco_path + '/' + row['image']
  print(image_path)

  base64_image = encode_image(image_path)

  payload = {
      "model": model,
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": user_prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 800
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  return response.json()

def _get_user_prompt(row, output_1_name: str = 'output_1', output_2_name: str = 'output_2'):
  conversations = row['conversations'][:-1]
  response_llava = row['conversations'][-1]['value']
  response_1 = row[output_1_name]['value']
  response_2 = row[output_2_name]['value']
  user_prompt = f"""
  [CONVERSATION CONTEXT with last turn being the question you need to answer and evaluate]: {conversations}, 
  [Candidate RESPONSE A to be evaluated]: {response_1}' 
  [Candidate RESPONSE B to be evaluated]: {response_2}'
  [Candidate RESPONSE C to be evaluated]: {response_llava}'
  """
  return user_prompt

def get_gpt_critiq(row, output_1_name: str = 'output_1', output_2_name: str = 'output_2'):
  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
  }
  conversations = row['conversations'][:-1]
  response_llava = row['conversations'][-1]['value']
  response_1 = row[output_1_name]['value']
  response_2 = row[output_2_name]['value']
  user_prompt = f"""
  [CONVERSATION CONTEXT with last turn being the question you need to answer and evaluate]: {conversations}, 
  [Candidate RESPONSE A to be evaluated]: {response_1}' 
  [Candidate RESPONSE B to be evaluated]: {response_2}'
  [Candidate RESPONSE C to be evaluated]: {response_llava}'
  """
  image_path = coco_path + '/' + row['image']
  print(image_path)

  base64_image = encode_image(image_path)

  payload = {
      "model": model,
      "messages": [
        {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": system_message
                }
            ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": user_prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 1800, 
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  return response.json()


def get_gpt_critiq_python_library(row, output_1_name: str = 'output_1', output_2_name: str = 'output_2'):
  # Assumes global variable of client
  user_prompt = _get_user_prompt(row, output_1_name, output_2_name)
  image_path = coco_path + '/' + row['image']
  print(image_path)

  base64_image = encode_image(image_path)

  response = client.with_options(max_retries=5).chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
          "role": "system",
          "content": [
              {
              "type": "text",
              "text": system_message
              }
          ]
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": user_prompt},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            },
          },
        ],
      }
    ],
    max_tokens=1800,
  )
  return response.model_dump_json()

def get_gpt4_comment(list_of_dict, output_path, start_index, end_index, output_1_name='output_1', output_2_name='output_2', time_sleep=15, existing_file={}):
    output_dict = {}
    if existing_file != {}:
        output_dict = existing_file
    # Update each dictionary and add to the output_dict
    excpetions_count = 0
    for index in tqdm(range(start_index, end_index)):
        if excpetions_count >= 2:
            print(f'Excpetions count {excpetions_count} is greater or equal 2. Exiting. I have reached the daily limit')
            break
        if isinstance(list_of_dict, dict):
            row = list_of_dict[str(index)].copy()
        elif isinstance(list_of_dict, list):
          row = list_of_dict[index].copy()
        try:
            gpt_response = get_gpt_critiq_python_library(row, output_1_name,output_2_name)
            row['gpt4v_response'] = gpt_response
            time.sleep(time_sleep)
            print(gpt_response)
            output_dict[index] = row
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            print(f'Sleeping for {time_sleep + 10} seconds')
            time.sleep(time_sleep + 10)
            print('Waking up and trying again')
            excpetions_count += 1

        # Save periodically to avoid losing all progress if the script stops
        if index % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(output_dict, f)

    # Write the final output to the file
    with open(output_path, 'w') as f:
        json.dump(output_dict, f)
    return output_dict

def check_openai_key(str):
    client = OpenAI(api_key=str)
    try:
        client.models.list()
    except openai.AuthenticationError as e:
        return False
    else:
        return True
class OutputPairs:
    def __init__(self, output_1, output_2):
        self.output_1 = output_1
        self.output_2 = output_2

vanilla_output = OutputPairs('resample_answer', 'origional_llava_response')
pair_of_llava_output = OutputPairs('output_1', 'output_2')
gpt4_genearted_output = OutputPairs('gpt4v_response', 'output_2')
if __name__ == "__main__":
    load_dotenv()
    shicheng_api = os.environ.get('shicheng_openai_api')
    alex_shengzhi_openai_api = os.environ.get('alex_shengzhi_openai_api')
    lisz1995 = os.environ.get('lisz1995')
    shengzhi_berkeley = os.environ.get('shengzhi_berkeley')
    westcliff = os.environ.get('westcliff')
    apple_id = os.environ.get('apple_id')
    exoplanet = os.environ.get('exoplanet')
    alexanderli = os.environ.get('alexanderli')

    environment_key = os.environ.get('OPENAI_KEY')

    api_lists = {
      'shicheng_api': shicheng_api,
      'environment_key': environment_key,
      'lisz1995': lisz1995,
      'alex_shengzhi_openai_api': alex_shengzhi_openai_api,
      'shengzhi_berkeley': shengzhi_berkeley,
      'westcliff': westcliff,
      'apple_id': apple_id,
      'exoplanet': exoplanet,
      'alexanderli': alexanderli
    }
    # Check key validity
    for api_key in api_lists.values():
        if check_openai_key(api_key):
            print(f"API key {api_key} is valid.")
        else:
            print(f"API key {api_key} is invalid.")
            raise ValueError(f"API key {api_key} is invalid.")
    json_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference.json'
    output_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_2125_2130.json'
    # Load the list of dictionaries from the JSON file
    with open(json_path, 'r') as file:
        list_of_dict = json.load(file)
    # try:
    #   list_of_dict = []
    #   with open(json_path, 'r') as file:
    #     for line in file:
    #       list_of_dict.append(json.loads(line))
    # except Exception as e:
    #   print(f"Error loading JSON file: {e}")
    #   # Handle the exception here

    output_names = pair_of_llava_output
    start_index = 3250
    length_of_api_lists = len(api_lists)
    end_index = start_index + 100 * length_of_api_lists
    combined_output_path = f'/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_subsample_gpt4v_response_{start_index}_{end_index}_combined.json'
    combined_output = {}
    for api_key_name, api_key in api_lists.items():
      client = OpenAI(api_key=api_key)
      output_path = f'/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_subsample_gpt4v_response_{start_index}_{start_index + 100}.json'
      if api_key_name in ['shicheng_api', 'environment_key']:
          continue # Skip the first two keys
          time_sleep = 2
      else:
          time_sleep = 20
      # I need to check if the file alrady exists, if so, i start from the last index
      existing_file = {}
      if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Loading the last index and continuing from there")
        with open(output_path, 'r') as f:
          existing_file = json.load(f)
          last_index = sorted([int(i) for i in list(existing_file.keys())])[-1]
          print(f"Last index is {last_index}, trying to start from {last_index} and finish the interval {start_index} to {start_index + 100}")
      output = get_gpt4_comment(list_of_dict, 
                                output_path, 
                                last_index, 
                                start_index + 100, 
                                output_1_name=output_names.output_1, 
                                output_2_name=output_names.output_2, 
                                time_sleep=time_sleep, 
                                existing_file=existing_file)
      # Only proceed to next 100 output if we finish the current 100
      # TODO: now if a key was used in trying to finish the 100 samples, it won't be used in the next 100 samples.
      if len(output) == 100:
        start_index += 100
        combined_output.update(output)
      else: 
        print(f"Current {api_key_name} is not working anymore. Skipping to next API key to try to finish the same 100 samples")
        continue
    # save combined ouptut
    with open(combined_output_path, 'w') as f:
        json.dump(combined_output, f)
