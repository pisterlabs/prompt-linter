import openai
import os
import logging
import time
import functions
import json

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("webdevgpt.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False
 
# Set up OpenAI API
client = openai.OpenAI()


# Set up OpenAI API - assistant IDs
coding_ai_id_35 = "asst_OO9fuCFGsDtmPhEWJ0Ihz7Zy"
prompt_ai_id_35 = "asst_B6yomAvgVF4dbtxJhYXp9Ok6"

prompt_ai = prompt_ai_id_35
coding_ai = coding_ai_id_35

def threadInit(function):
    logger.info("Initializing Thread")
    if function == "coding":
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "Follow the instructions to create a custom website.",
                }
            ]
        )
        logger.info("CodingAI Thread Initialized with ID: %s", thread.id)
    elif function == "prompt":
        thread = client.beta.threads.create()
        logger.info("PromptAI Thread Initialized with ID: %s", thread.id)
    
    return thread.id

codingThreadID = threadInit("coding")
promptThreadID = threadInit("prompt")
# This initilizes the message threads for coding and promtp AI

def functionCalling(run_object):
    #tool_calls = run_object.get("required_action", {}).get("submit_tool_outputs", {}).get("tool_calls", [])
    tool_calls = run_object.required_action.submit_tool_outputs.tool_calls
    
    for call in tool_calls:
        function_name = call.function.name
        arguments = call.function.arguments
        
        # Based on the function name, decide which Python function to call. Add more elif branches as you add more functions.
        if function_name == "create_website":
            functions.communications("Function", "create_website running.", "auto")
            arguments = json.loads(arguments)
            html_code = arguments.get("html_code", "")
            css_code = arguments.get("css_code", "")
            js_code = arguments.get("js_code", None) 

            # Call the create_website function with parameters extracted from the run_object.
            result = functions.create_website(html_code, css_code, js_code)
            #createdImage = functions.webpageImageRender()
            return result # return the result to the calling function.

        if function_name == "start_coder":
            functions.communications("Function", "start_coder running.", "auto")
            arguments = json.loads(arguments)
            message = arguments.get("message", "")
            modifedMessage = "PromptAI: " + message
            threadMessage(codingThreadID, modifedMessage)
            functions.communications("CodingAI", "Asking CodingAI to create a draft.", "auto")
            message = askCodingAI(codingThreadID) # CodingAI response is returned as a string.
            return message

def threadMessage(threadID, message="placeholder", action="create"):
    if action == "create":
        client.beta.threads.messages.create(thread_id=threadID, content=message, role="user")
        logger.info("Message: %s sent to thread %s", message, threadID)
        return "Message sent to thread"
    elif action == "createvis":
        client.beta.threads.messages.create(thread_id=threadID, content=message, role="user", visual=True)
        logger.info("Message: %s sent to thread %s", message, threadID)
        return "Message sent to thread"
    elif action == "createassist":
        client.beta.threads.messages.create(thread_id=threadID, content=message, role="assistant")
        logger.info("Message: %s sent to thread %s", message, threadID)
        return "Message sent to thread"
    elif action == "list":
        logger.info("Listing messages from thread")
        messages = client.beta.threads.messages.list(threadID)
        messageList = messages.data[0].content
        logger.info("Messages listed from thread")
        return messageList
    elif action == "newest":
        logger.info("Getting newest message from thread")
        messages = client.beta.threads.messages.list(threadID)
        newestMessage = messages.data[0].content[0].text.value
        logger.info("Newest message from thread retrieved")
        return newestMessage

def askCodingAI(threadID):
    logger.info("Asking CodingAI for a response")
    # 1. Create a run with the CodingAI assistant thread.
    run = client.beta.threads.runs.create(thread_id=threadID, assistant_id=coding_ai)
    # 2. Poll the Assistants API for a completed response from an assistant run
    while True:
        logger.info("CodingAI response status: %s", run.status)
        if run.status == 'completed':
            # 4. Get the final message response from the run.
            codingAIresponse = threadMessage(threadID=threadID, action="newest") # This gets the newest message from the thread.
            functions.communications("CodingAI", codingAIresponse, "comm")
            logger.info("CodingAI response: " + codingAIresponse)
            return codingAIresponse
        if run.status == 'requires_action':
            functions.communications("CodingAI", "Generating the website.", "auto")
            logging.info("CodingAI response requires action")
            # 3. The CodingAI assistant has a required action. In this case, it is a tool call.
            runId = run.id
            toolId = run.required_action.submit_tool_outputs.tool_calls[0].id
            # 4. This gets the runID and the toolID so that it can be used to update the action once the "websiteCreation" function is called and the image of the rendered website is returned.
            logger.info("Tool ID: {} and Run ID: {}".format(toolId, runId))
            websiteCreationResponse = functionCalling(run) # This calls the function that will create the website and return the image of the rendered website.
            logger.info("Website creation response: " + websiteCreationResponse)
            client.beta.threads.runs.submit_tool_outputs(
                run_id=runId,
                thread_id=threadID,
                tool_outputs=[
                    {
                        "tool_call_id": toolId,
                        "output": "success",
                        # This will submit a tool (update) with the toolID and the output will be a link (string) to the image of the rendered website. This will be used to update the action.
                    }
                ]
            )
            logging.info("CodingAI response action submitted")
            time.sleep(.5)  # wait for 1 second before checking the status again
            run = client.beta.threads.runs.retrieve(run_id=run.id, thread_id=threadID)
        else:
            time.sleep(.5)  # wait for 1 second before checking the status again
            run = client.beta.threads.runs.retrieve(run_id=run.id, thread_id=threadID)
    
def askPromptAI(threadID):
    logger.info("Asking PromptAI for a response")
    # 1. Create a run with the PromptAI assistant thread.
    run = client.beta.threads.runs.create(thread_id=threadID, assistant_id=prompt_ai)
    # 2. Poll the Assistants API for a completed response from an assistant run
    while True:
        logger.info("PromptAI response status: %s", run.status)
        if run.status == 'completed':
            # 3. The PromptAI assistant has completed and returned a response.
            # 4. Get the final message response from the run.
            promptAIresponse = threadMessage(threadID=threadID, action="newest")
            functions.communications("PromptAI", promptAIresponse, "comm")
            logger.info("Received completed response from CodingAI | >>> " + promptAIresponse)
            break
        if run.status == 'requires_action':
            logger.info("PromptAI response requires action")
            # Get the action prompt from the run
            runId = run.id
            toolId = run.required_action.submit_tool_outputs.tool_calls[0].id
            logger.info("Tool ID: {} and Run ID: {}".format(toolId, runId))
            responseFromCodingAI = functionCalling(run)
            
            client.beta.threads.runs.submit_tool_outputs(
                run_id=runId,
                thread_id=threadID,
                tool_outputs=[
                    {
                        "tool_call_id": toolId,
                        "output": responseFromCodingAI
                    }
                ]
            )
            logger.info("PromptAI response action submitted")
            time.sleep(1)  # wait for 1 second before checking the status again
            run = client.beta.threads.runs.retrieve(run_id=run.id, thread_id=threadID)
            if run.status == 'completed':
                modifiedResponse = "CodingAI: " + responseFromCodingAI
                threadMessage(threadID=threadID, message=modifiedResponse, action="create")
        else:
            time.sleep(1)  # wait for 1 second before checking the status again
            run = client.beta.threads.runs.retrieve(run_id=run.id, thread_id=threadID)
    
def threadObjectTestRetrevial(threadID):
    logger.info("Retrieving thread object")
    threadObject = client.beta.threads.messages.list(threadID)
    n = 0
    for item in threadObject.data:
        print("-------------------------------------------------------------\nThread Object nth: {} | Role {}: ".format(n, item.role) + str(item.content[0].text.value))
        n += 1
    #print("Thread Object: " + str(threadObject.data[0].content[0].text.value))
    
    logger.info("Thread object retrieved")
    return threadObject

def convertThreadToMessageList(threadID, systemMessage="You are a helpful assistant."):
    logger.info("Converting thread object to message list")
    threadObject = client.beta.threads.messages.list(threadID)
    messageList = [
        {"role": "system", "content": systemMessage}
    ]

    """
    Reference for Vision API
    content=[
        {"type": "text", "text": "What’s in this image?"},
        {
            "type": "image_url",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        },
    ]
    """
    for item in threadObject.data:
        role = item.role
        message = item.content[0].text.value
        messageList.append({"role": role, "content": message})
    logger.info("Thread object converted to message list")
    return messageList

def chatPromptAIVision(threadID, imageBase64=None, comments=None):
    messageList = convertThreadToMessageList(threadID, "You know the original website prompt and request. If you think that the website that CodingAI created does not meet your original request, then format your response like this: REVISE|<comments>. If you think that the website that CodingAI created does meet your original request, then format your response like this: NOREVISE|<comments>. You MUST respond with either REVISE or NOREVISE.")
    if imageBase64 != None and comments != None:
        messageList.append({"role": "user", "content": [
                    {
                        "type": "text", 
                        "text": comments
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{imageBase64}"
                        }
                    }
                ]
            }
        )
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messageList,
        max_tokens=300,
        
    )
    # Check if a function was called
    revisedWebsite = False
    visionResponse = response.choices[0].message.content
    if visionResponse.startswith("REVISE"):
        # If a function was called, execute the function and return the result    
        comments = visionResponse.split("|")[1]
        revisedWebsite = True            
        functions.communications("PromptAI", "Asking CodingAI to revise the website. PromptAI did not find the website satisfactory.", "auto")
    else:
        comments = visionResponse.split("|")[1]
        functions.communications("PromptAI", comments, "comm")

    return revisedWebsite, comments

def main(userPrompt):
    logger.info("Starting main function")
    functions.communications("PromptAI", "Hello, I am PromptAI. I am here to help you create a website.", "auto")

    # userPrompt = "Create a website for Ryan Vogel that is modern with white bold text and a darker background color." #input("What would you like the website to look like? ")
    # userPrompt = input("Describe in detail what you would like the website to function with and do.\nUser: ")
    iteration = 0
    threadMessage(threadID=promptThreadID, message=userPrompt, action="create")
    # 1. Add the users prompt to the PromptAI thread
    askPromptAI(promptThreadID)
    # 2. Ask the PromptAI for a response
    while iteration < 3:
        functions.communications("PromptAI", "Asking PromptAI to check the work of CodingAI, this is draft {}.".format(iteration + 1), "auto")
        pictureFilename = functions.webpageImageRender()
        functions.communications("WEBPIC", pictureFilename, "comm")
        # 3. Render the webpage, save it as an image, and return the image
        base64Image = functions.convertImageFileToBase64String(pictureFilename)
        # 4. Pass the base64 image to the Vision assistant to evaluate if it is sufficient to the original prompt.
        revisonNeeded, visionResponse = chatPromptAIVision(promptThreadID, base64Image, "I have attached a screenshot of the website that CodingAI created. Does this attached image of the website meet the original users request? If it does not then call the revise function, but if it does then just return with your final remarks (keep them consise).")
        # 5. If the image is not sufficient, then ask the user for revisions.
        if revisonNeeded:
            revisedResponse = "I do not think that the website you created meets the original request. Please revise the website with the following comments: " + visionResponse
            threadMessage(threadID=codingThreadID, message=revisedResponse, action="create")
            # 6. Add the users revisions to the PromptAI thread
            codingAIResponse = askCodingAI(promptThreadID)
            iteration += 1
        else:
            threadMessage(threadID=promptThreadID, message=visionResponse, action="create")            
            break

        # 4. Convert the image to a base64 string
    logger.info("Main function completed")


if __name__ == "__main__":
    main()
    # threadMessage(threadID=promptThreadID, message="Hello", action="create")
    # askPromptAI(promptThreadID)
    # threadMessage(threadID=promptThreadID, message="How are you?", action="create")
    # askPromptAI(promptThreadID)
    # print("Prompt Thread: " + promptThreadID)
    # threadObjectTestRetrevial(promptThreadID)
    # threadMessage(threadID=promptThreadID, message="Give me a fun fact", action="create")
    # print(chatPromptAIVision(promptThreadID))
    

