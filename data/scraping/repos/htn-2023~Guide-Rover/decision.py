import config as config, json

# import and configure OpenAI
import openai
import cohere
# import and configure asyncio and Interactive Brokers ib_insync package
import asyncio
import nest_asyncio
nest_asyncio.apply()


last_occurrence = -1

async def check_transcript():
    global last_occurrence

    with open('communication/transcriptions/transcript.txt') as f:
        text = f.read()
        occurrence = text.lower().rfind(config.COMMAND_WORD)
        if occurrence != last_occurrence:
            print(f"Found new {config.COMMAND_WORD} at {occurrence}")

            # get command starting at occurrence of command word
            command = text[occurrence:]
            print(command)

            # store last occurrence so we don't repeat the same command
            last_occurrence = occurrence

            prompt = f"""{config.PROMPT_INSTRUCTIONS}
            {command}

            {config.PROMPT_OUTPUT_FORMAT}
            """

            print(prompt)
            path = 'gpt'
            if path == 'gpt':
                openai.api_key = config.OPENAI_API_KEY
                engine = 'gpt-4'
                response = openai.ChatCompletion.create(
                    model=engine,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    frequency_penalty=0.0
                )
            elif path == 'cohere':
                co = cohere.Client('kAbNlTv4YeZHafQ2ynhWeKYcpAkWbBoZC94vqT4E')
                response = co.chat(
                prompt,
                temperature=0.8,
            )
            try:
                print('-----------------------------------')
                print(response['choices'][0]['message']['content'].strip())
                global response_dict
                response_dict = json.loads(response['choices'][0]['message']['content'].strip())
                print(response_dict)
                print('-----------------------------------')
            except Exception as e:
                print(f"error parsing response from OpenAI {e}")
                return

async def run_periodically(interval, periodic_function):
    while True:
        await asyncio.gather(asyncio.sleep(interval), periodic_function())

asyncio.run(run_periodically(1, check_transcript))
