from phue import Bridge
import openai
import dotenv
import time
import json
import logging
from audio import get_question
from pydub import AudioSegment
from pydub.playback import play
import io

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Initialize OpenAI Client
client = openai.Client()

# Initialize Hue Bridge
ip = "192.168.0.211"
b = Bridge(ip)
b.connect()

def pretty_print(messages):
    print("# Messages")
    last_message = None  # Initialize last_message to None
    for i, m in enumerate(messages):
        last_message = m  # Update last_message with the current message
        print(f"{m.role}: {m.content[0].text.value}")

    if last_message:
        # Process the last message outside the loop
        speech = client.audio.speech.create(
            model="tts-1",
            voice="nova",  # Fixed typo in "voice" attribute
            input=last_message.content[0].text.value)
        # speech contains an MP3, play it
        audio = AudioSegment.from_mp3(io.BytesIO(speech.content))
        play(audio)

    print()


def wait_for_run(run):
    while run.status == 'queued' or run.status == 'in_progress':
        run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
        )
        time.sleep(0.5)

    return run

def contains_error(data):
    if isinstance(data, dict):
        if 'error' in data:
            return True
        for key in data:
            if contains_error(data[key]):
                return True
    elif isinstance(data, list):
        for item in data:
            if contains_error(item):
                return True
    return False

def get_lights(location):
    # location not used for now
    lights = b.get_light_objects('name')

    # lights is a dict with light name as key, return list of keys
    return json.dumps(list(lights.keys()))

def set_light(light, state):
    response = b.set_light(light, 'on', state)
    if contains_error(response):
        log.error("Error setting light")
        return '{"success": False}'
    
    log.info(f"Light on? {state}")
    return '{"success": True}'
    
def set_light_brightness(light, brightness_percent):
        # brightness is a percentage of 254 rounded up
        brightness = round(brightness_percent / 100 * 254)
        response1 = b.set_light(light, 'on', True)
        response2 = b.set_light(light, 'bri', brightness)
        if contains_error(response1) or contains_error(response2):
            log.error("Error setting light brightness")
            return '{"success": False}'
        
        log.info("Light brightness set successfully")
        return '{"success": True}'

assistant_id = 'asst_12rntJWxpK1jPHAlPazTcThi' # Hue assistant id in OpenAI

# create a thread
thread_id = client.beta.threads.create().id
log.info(f"Created thread with id {thread_id}")

# keep asking for user input until user types 'exit'
while True:
    # user_input = input("You: ")

    # press enter to start recording
    input("Press enter to start recording")
    user_input = get_question()
    

    # add message to thread
    client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input
    )
    log.info(f"Added message to thread {thread_id}")

    # run the assistant
    run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions="""
            Please address the user as Geert. Help the user with Hue lights. 
            When the user says 'floor lamp' or similar, only drop the word lamp. 
            Always capitalize the lamp name. Never try an action again unless the user asks."""
    )
    log.info(f"Created run with id {run.id}")

    # wait for run
    run = wait_for_run(run)

    run_json = run.model_dump()

    # write run_json to file
    with open('run.json', 'w') as f:
        json.dump(run_json, f, indent=4)
        log.info("Wrote run.json to file")

    if run.required_action:
        # does this run require action; we just take first one
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        log.info(f"Detected tool calls: {tool_calls}")

        # For each tool call
        tool_outputs = []
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # print function and arguments
            log.info(f"Processing function name: {func_name}")
            log.info(f"Function arguments: {arguments}")

            if func_name == 'set_light':
                result = set_light(arguments['light'], bool(arguments['state']))
            elif func_name == 'set_light_brightness':
                result = set_light_brightness(arguments['light'], arguments['brightness'])
            elif func_name == 'get_lights':
                result = get_lights(arguments['location'])
            else:
                continue

            # Add the tool output to the list
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": json.dumps(result)
            })

        # present tool outputs to assistant
        run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
        )
        log.info("Submitted tool outputs")

        run = wait_for_run(run)
        log.info("Tool output run completed")
    

    # print responses
    messages = client.beta.threads.messages.list(
        thread_id=thread_id, limit=1, order='asc'
    )
    pretty_print(messages)