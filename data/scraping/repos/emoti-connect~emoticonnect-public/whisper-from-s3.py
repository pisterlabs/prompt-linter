import boto3
import os
import openai
import json

openai.api_key = os.environ['openai_key']

def chatgpt_inference(messages, model="gpt-3.5-turbo", max_tokens=256, temperature=0.2):
  response = openai.ChatCompletion.create(
      model=model,
      max_tokens=max_tokens,
      temperature=temperature,
      messages=messages,
  )
  return  response


def whisper_inference(media_file_path):
    API_KEY = '[openai_api_key]'
    model_id = 'whisper-1'

    media_file_path = '/tmp/audio.mp3'
    media_file = open(media_file_path, 'rb')

    response = openai.Audio.transcribe(
        api_key=API_KEY,
        model=model_id,
        file=media_file
    )

    responsetext = response["text"]    
    return responsetext
  
  

def lambda_handler(event, context):
    # Specify the AWS credentials and region
    aws_access_key_id = '[aws_key]'
    aws_secret_access_key = '[aws_secret]'
    aws_region = 'us-east-1'

    # Specify the bucket name and object key
    bucket_name = 'emoticonnect'
    object_key = 'audio/output/processed-audio.mp3'
    local_file_name = '/tmp/audio.mp3'  # Replace with the desired name and path

    # Create a Boto3 S3 client
    s3_client = boto3.client('s3', region_name=aws_region, aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key)

    # Download the file from the S3 bucket
    s3_client.download_file(bucket_name, object_key, local_file_name)

    output = whisper_inference('/tmp/audio.mp3')

                               
                               
    if("key" in event and event["key"]=="0987654321"):
        model = "gpt-3.5-turbo"
        messages = json.loads('[{"role": "user","content":"Say nothing."}]')
        max_tokens = 256      
        temperature=0.3

        if("model" in event):
            model = event["model"]

        if("apple_watch_text" in event):
            messages = json.loads("""[{
        "role": "system",
        "content": "We are taking speech from one person and converting it into emotion for neurodivergent individual to understand the emotions taking place in the conversation (use this context to derive emotions from the text for the neurodivergent person)"
        },{
        "role": "user",
        "content": 
        "Emotion: Happiness\\nHappiness Intensity: [ChatGPTs assesment of happiness intensity on a scale of 1-10]\\nHappiness Confidence: [ChatGPTs confidence level for detecting happiness as a percentage]\\n\\nEmotion: Sadness\\nSadness Intensity: [ChatGPTs assesment of sadness intensity on a scale of 1-10]\\nSadness Confidence: [ChatGPTs confidence level for detecting sadness as a percentage]\\n\\nEmotion: Anger\\nAnger Intensity: [ChatGPTs assesment of angerintensity on a scale of 1-10]\\nAnger Confidence: [ChatGPT confidence level for detecting anger as a percentage]\\n\\nEmotion: Fear\\nFear Intensity: [ChatGPTs assesment of fear intensity on a scale of 1-10]\\nFear Confidence: [ChatGPSs confidence level for detecting fear as a percentage]\\n\\nEmotion: Neutral\\nNeutral Intensity: [ChatGPTs assesment of neutralintensity on a scale of 1-10]\\nNeutral Confidence: [ChatGPTs confidence level for detecting neutrality as a percentage]\\n\\nPrimary Emotion: [ChatGPTs calculation of the most proeminent emotion based on confidence and intensity of the five emotions]\\nSecondary Emotion: [ChatGPTs calculation of the second most proeminent emotion based on confidence and intensity of the five emotions]\\nBalanced or Not Balanced: [ChatGPTs assesment whether the primary and secondary emotions are balanced or not balanced (1 means balanced, 0 means not balanced)]"
        },{
        "role": "user",
        "content": "Can you provide an answer in the form of json of 'Primary Emotion' and 'confidence' score for the primary emotion for the following text:"
        }]""")
            messages[2]["content"] += output
        
    
        if("messages" in event):
            messages = event["messages"]

        if("max_tokens" in event):
            max_tokens = event["max_tokens"]
        
        if("temperature" in event):
            temperature = event["temperature"]
        
        response = chatgpt_inference(messages = messages, model = model, max_tokens=max_tokens, temperature=temperature)                               

        if response and "choices" in response:
            assistant_message = response["choices"][0]["message"]["content"]
            assistant_data = json.loads(assistant_message)

            primary_emotion = assistant_data.get("Primary Emotion")
            confidence_score = assistant_data.get("Confidence")

            response = {
                "Primary Emotion": primary_emotion,
                "Confidence Score": confidence_score
            }
    
    return {
        'statusCode': 200,
        'output': output,
        'response': response,
        'body': 'Processing completed successfully.'
    }        
