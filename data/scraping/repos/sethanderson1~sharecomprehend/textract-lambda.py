import json
import boto3
import openai
import urllib.parse

ora = open('openairesp.json',)
openai_response_dict = json.load(ora)


def lambda_handler(event, context):
    textract = boto3.client("textract")

    if event:
        file_obj = event["Records"][0]
        bucketname = str(file_obj["s3"]["bucket"]["name"])
        filename = urllib.parse.unquote_plus(str(file_obj["s3"]["object"]["key"]), encoding='utf-8')

        print(f"Bucket: {bucketname} ::: Key: {filename}")
        
        base_dict = { 'bucket' : bucketname, 'file' : filename }
    
        response = textract.detect_document_text(
            Document= {
                "S3Object": {
                    "Bucket": bucketname,
                    "Name": filename,
                }
            }
        )

        raw_text = extract_text(response, extract_by="LINE")
        print(f'RAW_TEXT: {raw_text}')
        
        openai_response = openai_handler(raw_text)
        
      
        line_text_dict = {'document-lines' : raw_text}
        base_dict.update(openai_response)
        base_dict.update(line_text_dict)
        print(f'base_dict: {json.dumps(base_dict)}')
        
    return {
        'statusCode': 200,
        'body': json.dumps(base_dict)
    }
    
    
def extract_text(response, extract_by="LINE"):
    blocks = response['Blocks']
    
    line_text = []
    for block in response["Blocks"]:
        if block["BlockType"] == extract_by:
            line_text.append(block["Text"])
        
    return line_text
    
def openai_handler(questionpart):
    #openai.api_key = 'xxxxxxxxxxxxx'
    questionpartstring = ' '.join(map(str, questionpart))
    question = 'top 10 tags with detail for this text ' + questionpartstring
        
    print(f'question: {question}')

    """
    response = openai.Completion.create(
        model = "text-davinci-003",
        prompt = question,
        temperature = 1,
        max_tokens = 100)
       
      """ 
    #NOTE: THIS IS TO ASSIGN THE STATIC JSON AS OPPOSED TO THE OPENAI RESPONCE COMMENTED ABOVE  
    response = openai_response_dict
    print(f' openai-dict response: {response}')

    return (response)
    
    """
    return {
        'statusCode': 200,
        'body': response['choices'][0]['text']
    }

A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
    """
