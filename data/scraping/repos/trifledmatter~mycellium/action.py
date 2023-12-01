import config, json, openai, asyncio, nest_asyncio, os
from ib_insync import *
from datetime import datetime as dt

nest_asyncio.apply()
openai.api_key = config.OPENAI_API_KEY

ib = IB()
ib.connect('127.0.0.1', config.IB_PORT, clientId=1)

last_occurance = -1

async def get_last_occurance():
    # Check the transcript
    global last_occurance

    with open(f"{config.SERVICE_DIRECTORY}/transcript.txt") as f:
        content = f.read()
        occurance = content.lower().rfind(config.COMMAND_WORD)

        if occurance != last_occurance:
            # Found a new occurance of the command word
            command = content[occurance:]
            last_occurance = occurance

            prompt = f"""{config.INSTRUCTIONS}
            {command}

            {config.OUTPUT_FORMAT}
            """

            engine = config.OPENAI_ENGINE
            temperature = config.OPENAI_TEMPERATURE
            max_tokens = config.OPENAI_MAX_TOKENS
            top_p = config.OPENAI_TOP_P
            frequency_penalty = config.OPENAI_FREQUENCY_PENALTY
            presence_penalty = config.OPENAI_PRESENCE_PENALTY

            r = await openai.completions.create(
                engine,
                prompt,
                temperature,
                max_tokens,
                top_p,
                frequency_penalty,
                presence_penalty
            )

            # We'll store the responses in a folder
            response = r['choices'][0]['text']
            response_as_dictionary = json.loads(r['choices'][0]['text'].strip())

            # We'll save the response to a file
            filename_base = "response_" + str(dt.now().strftime("%Y%m%d-%H%M%S"))
            filename = filename_base + ".txt"

            if not os.path.exists(config.MODEL_RESPONSE_DIRECTORY):
                os.makedirs(config.MODEL_RESPONSE_DIRECTORY)

            with open(os.path.join(config.MODEL_RESPONSE_DIRECTORY, filename), "w") as f:
                f.write(response)
            
            # Parse the response
            try:
                action = response_as_dictionary['action']
                request = response_as_dictionary['request']
                quantity = response_as_dictionary['quantity']
                item = response_as_dictionary['item']

            except Exception as e:
                print(f"There was an error parsing the reponse from OpenAI")
                return
            
            print_format = f"""
            Item: {item}
            Action: {action}
            Request: {request}
            Quantity: {quantity}

            Timestamp: {dt.now()}
            """
            
            print(print_format)

            
async def run_periodically(interval, func):
    while True:
        await asyncio.gather(asyncio.sleep(interval, func()))

asyncio.run(run_periodically(1, get_last_occurance))
ib.run()