import asyncio
import openai
from gtts import gTTS
from playsound import playsound
import os
import tempfile

TOKEN_LIMIT = 4096
CHARS_PER_TOKEN = 4  # approximate


def parse_commands(text: str):
    commands = []
    command = ""
    in_quotes = False
    for char in text:
        if char == '"' and not in_quotes:
            in_quotes = True
        elif char == '"' and in_quotes:
            in_quotes = False
        elif char == ';' and not in_quotes:
            commands.append(command.strip())
            command = ""
        else:
            command += char
    if command:
        commands.append(command.strip())
    return commands


# Stubbed bot functions that mimic the real ones but do nothing
async def move(distance):
    print(f"[move] [{distance}]")
    await asyncio.sleep(1)


async def turn(degrees):
    print(f"[turn] [{degrees}]")
    await asyncio.sleep(1)


async def wait(seconds):
    print(f"[wait] [{seconds}]")
    await asyncio.sleep(seconds)


async def play_audio(file_path):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, playsound, file_path)


async def speak(tts_queue, text_to_speak):
    print(f"[speak] [{text_to_speak}]")
    tts = gTTS(text=text_to_speak, lang='en')
    file_path = os.path.join(tempfile.gettempdir(), f"tts_{hash(text_to_speak)}.mp3")
    tts.save(file_path)
    await tts_queue.put(file_path)


class Bot:
    async def handle_tts_queue(self):
        while True:
            file_path = await self.tts_queue.get()
            if file_path is None:
                break
            await play_audio(file_path)
            os.remove(file_path)

    def reset_state(self):
        # Wait for the previous handle_tts_task to finish, if any
        if self.handle_tts_task is not None:
            asyncio.ensure_future(self.handle_tts_task)

        # Clear the tts_queue and create a new handle_tts_task
        self.tts_queue = asyncio.Queue()
        self.handle_tts_task = asyncio.create_task(self.handle_tts_queue())

    def __init__(self):
        self.tts_queue = asyncio.Queue()
        self.handle_tts_task = None
        self.reset_state()  # Call reset_state without 'await' here


async def get_chunk(generator):
    try:
        chunk = next(generator)
    except StopIteration:
        return None
    await asyncio.sleep(0)  # Yield control to the event loop
    return chunk


def count_tokens(messages):
    counter = 0.0
    for message in messages:
        counter += len(message) / CHARS_PER_TOKEN
    return int(round(counter))


def summarize(message, model='gpt-3.5-turbo', print_during=False):
    messages = [{"role": "user", "content": f"Please summarize the following message in a sentence: \"{message}\""}]

    full_message = ""
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=True
    )

    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']
        text = chunk_message.get('content', '')
        full_message += text
        if print_during:
            print(text)
        yield text

    return full_message


def ensure_within_token_limit(messages):
    while count_tokens(messages) >= TOKEN_LIMIT:
        message_to_remove = messages.pop(0)
        summary = summarize(message_to_remove['content'])
        messages.insert(0, {'role': message_to_remove['role'], 'content': summary})


class ChatCompletion:
    main_system_prompt = """
You are an AI connected to a Robot. 
You can run one or more commands when asked.
Translate the user's natural language to commands.

Your messages are in the format:
command_1(argument_1_1,...,argument_1_n);command_2(argument_2_1,...,argument_2_n);command_m(argument_m_1,...,argument_m_n)

Commands are separated by ';', arguments are separated by ','.
Don't split on anything inside double quotes because that's a string argument.

Commands include: 
1. move(distance_cm) (BLOCKING CALL)
2. speak(text_to_speak) (NON-BLOCKING CALL)
3. turn(degrees) (BLOCKING CALL)
4. wait(seconds) (BLOCKING CALL)

Example input:
Draw a 10cm x 10cm square while saying a haiku about robots.
Example output:
"""+'speak("Robots work hard all day. Making life much easier. For us humans too");' \
        'move(10.0);turn(90.0);move(10.0);turn(90.0);move(10.0);turn(90.0);move(10.0);turn(90.0);speak("done!") '

    def __init__(self, api_key, bot):
        self.api_key = api_key
        self.bot = bot
        self.conversation_history = [{'role': 'system', 'content': self.main_system_prompt}]
        openai.api_key = self.api_key

    def get_generator(self, prompt: str, model: str = 'gpt-3.5-turbo', temperature: float = 0.8):
        self.conversation_history += [{'role': 'user', 'content': prompt}]

        ensure_within_token_limit(self.conversation_history)

        generator = openai.ChatCompletion.create(
            model=model,
            messages=self.conversation_history,
            temperature=temperature,
            stream=True
        )
        return generator

    async def read_and_enqueue_commands(self, prompt, queue, print_text=False):
        generator = self.get_generator(prompt)

        while True:
            chunk = await get_chunk(generator)
            if chunk is None:
                break

            chunk_message = chunk['choices'][0]['delta']
            text = chunk_message.get('content', '')
            if print_text:
                print(text, end='')  # Print the text as it comes in
            await queue.put(text)

        await queue.put(None)  # Add sentinel value to indicate the end of the stream

    async def execute_command_list(self, command_list):
        for cmd_str in command_list:
            try:
                cmd, args_str = cmd_str.split('(', 1)
                args = args_str.rstrip(')').split(',')
                if cmd == "move":
                    await move(float(args[0]))
                elif cmd == "speak":
                    await speak(self.bot.tts_queue, args[0].strip('"'))
                elif cmd == "turn":
                    await turn(float(args[0]))
                elif cmd == "wait":
                    await wait(float(args[0]))
            except Exception as e:
                print(f"[error] {e} [command] {cmd_str}")

    async def execute_commands(self, queue):
        command_text = ""
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            command_text += chunk
            if ';' in command_text:
                commands, remainder = command_text.rsplit(';', 1)
                command_list = parse_commands(commands)
                await self.execute_command_list(command_list)
                command_text = remainder

        # Execute the remaining commands after the end of the stream
        command_list = parse_commands(command_text)
        await self.execute_command_list(command_list)


async def main():
    bot = Bot()
    with open('apikey.txt', 'r') as f:
        api_key = f.read().strip()
    chat_completion = ChatCompletion(api_key, bot)

    while True:
        prompt = input("Enter prompt: ")
        queue = asyncio.Queue()  # Clear the queue by creating a new instance
        bot.reset_state()  # Call reset_state without 'await' here
        task1 = asyncio.create_task(chat_completion.read_and_enqueue_commands(prompt, queue))
        task2 = asyncio.create_task(chat_completion.execute_commands(queue))
        await asyncio.gather(task1, task2)
        await bot.tts_queue.put(None)  # Add sentinel value to indicate the end of the stream

if __name__ == "__main__":
    asyncio.run(main())
