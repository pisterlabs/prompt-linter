import azure.functions as func
import logging


#openai variables
import openai
from azure.identity import DefaultAzureCredential

azure_credential = DefaultAzureCredential()
ad_token = azure_credential.get_token("https://cognitiveservices.azure.com/.default")

app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", path="emails",
                               connection="AzureWebJobsStorage")
@app.blob_output(arg_name="outputblob", path="output/{rand-guid}",
                               connection="AzureWebJobsStorage")
def caterpillar_function(myblob: func.InputStream, outputblob: func.Out[str]):
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {myblob.name}"
                f"Blob Size: {myblob.length} bytes")
    
    content = myblob.read().decode('utf-8')
    logging.info(f"Blob content: {content}")
    openaiprompt1 = "Parse this email object and analyse the sentiment and topic of the email body and generate a json response message."
    openaiprompt2 = "The sendersEmotion should be selected from one of the following range of emotions (Appreciative, Concerned, Frustrated, Supportive, Angry, Pleased, Worried, Thankful, Dissatisfied, Hopeful). The only response I require is the original json message in the exact same format but update the json message populating the 'topic', 'sentiment', 'requiresResponse', 'suggestedResponse', 'containsSensitiveData', 'confidential', sendersEmotion' properties based on the analysis of the content. Where a response is required provide a polite suggested response. Do not provide any text in the response other than the json message which should be provided in the following format:"
    openaiprompt3 = '{/"new/": true,/"sender/": /"aPerson@email.com/",/"received/": /"31/10/2023/",/"hasAttachments": true,/"attachments/": {/"1": /"myphoto.jpeg/",/"2": /"myletter.docx/",},/"subject/": /"Feedback relating to the hack session/",/"body/": /"<p>Dear Paul,</p><p>This is a sample email to reach out to tell you im very happy with all the sweets and coffee available at the sponsored hack session.</p><p>Regards</p><p>Nick (hacker)</p>/",/"topic/": hackathon/"sentiment": feedback/"requiresResponse": false,/"suggestedResponse": /"Hi Nick, thank you for your email, im glad you enjoyed the session. Regards Paul/",/"containsSensitiveData/": false,/"confidential/": false,/"sendersEmotion/" /"Appreciative/"}'

    openairequest = " ".join([openaiprompt1, content, openaiprompt2, openaiprompt3])
    logging.info(f"OpenAI request: {openairequest}")

    apim_gateway_url = 'https://genaiapim.azure-api.net' 
    team_name = 'incognito-insects' # For example if your Team Name is cocreate-cumbersome-cat this would be team_name = 'cumbersome-cat'
    api_base = f'{apim_gateway_url}/{team_name}' 

    openai.api_type = 'azure_ad'
    openai.api_key = ad_token.token
    openai.api_base = "https://genaiapim.azure-api.net/magnificent-monkeys"
    openai.api_version = '2023-03-15-preview'
    engine = 'gpt-4-32k'

    print(f'Currently using the following OpenAI endpoint: {openai.api_base}')
    print(f'Currently using the folllowing OpenAI API version: {openai.api_version}') 
    print(f'Currently targetting the following OPENAI model: {engine}')

    messages = [{'role': 'user', 'content': openairequest}]

    full_openai_chat_response = openai.ChatCompletion.create(engine=engine, messages=messages)

    if not full_openai_chat_response.choices[0].message.content:
        print('OpenAI response was empty.')

    content = full_openai_chat_response.choices[0].message.content

    # only get { ... } part of the response
    responsecontent = content[content.find('{'):content.rfind('}')+1]

    print(f'OpenAI response: {responsecontent}')

    outputblob.set(responsecontent)

    


