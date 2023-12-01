import openai
import time
from tools.functionsConfigurator import functionsConfigurator as functionsConfigurator
from tools.openAiHelper import openAiHelper as openAiHelper

openAiListenerObj = openAiHelper()
functionConfigHelper = functionsConfigurator()
#raise SystemExit

tools_list = [
    functionConfigHelper.getRecomendedDestinationConfiguration()
]


# Initialize the client
client = openai.OpenAI()
print("Create Assistant")

# Step 1: Create an Assistant

newAssistant = 1
newThread = 1
assistant_id = "asst_PWXcAjzaPa8GaNnOTT5DD31f"
threadId = "thread_ufoLhtd06azdjhdTQIGDhmmz"
messageToSend = "I am traveling to Rome"


if newAssistant == 1:
    assistant = client.beta.assistants.create(
        name="Recomended hotel assistant",
        instructions="you are a travel agent who recomended for hotels in a destination",
        tools=tools_list,
        model="gpt-4-1106-preview",
    )
    assistant_id = assistant.id
    print("Assistant ID : ",assistant_id)


if newThread == 1:
    # Step 2: Create a Thread
    thread = client.beta.threads.create()
    threadId = thread.id
    print("Thread ID : ",threadId)

messageToSend = input("User: ")
while messageToSend != "exit":
    run = openAiListenerObj.sendMessage(client,assistant_id, threadId, messageToSend)
    #print(run.model_dump_json(indent=4))
    openAiListenerObj.listen(client, assistant_id, threadId, run)
    messageToSend = input("User: ")
raise SystemExit
