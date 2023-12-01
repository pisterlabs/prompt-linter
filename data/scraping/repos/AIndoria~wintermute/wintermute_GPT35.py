last_two_results = []
contexts = {}
import irc.bot
import irc.strings
import irc.client
import random
import datetime
import time
import sqlite3
import requests
import openai
import string
import textwrap
import re
last_message_time={}
password = "password"
system_prompt = "You are a semi-helpful but weird and quirky assistant. You have opinions. Sometimes you refuse to help and just make up a weird reason as to why."
messages_dict = {}
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
openai.api_key = "YOUR_OPENAI_API_KEY_HERE"


def gpt3_request():
    # Set the model engine to use
    model_engine = "text-davinci-003"
    max_tokens = 500
    temperature = 0.5
    top_p = 1
    frequency_penalty = 0.5
    presence_penalty = 0.5

    # Make the API request
    response = openai.Completion.create(
      engine=model_engine,
      prompt="Make up an error message blaming either the OpenAI, their servers, or their team members for the error. Be creative and insulting.",
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty,
    )
    result = response["choices"][0]["text"]
    print(response)
    return result




def openai_api_request(input_text, channel):
    # Set the model engine to use
    messages = messages_dict.setdefault(channel, [{"role": "system", "content": system_prompt}])
    current_time = time.time()
    if channel in last_message_time and (current_time - last_message_time[channel]) > (30 * 60):
        # If 30 minutes have elapsed, clear all messages for this channel
        messages.clear()
        messages = messages_dict.setdefault(channel, [{"role": "system", "content": system_prompt}])
    if len(messages) >= 4:
        messages.pop(0)
    messages.append({"role": "user", "content": input_text})
    model_engine = "gpt-3.5-turbo"
    max_tokens = 588
    temperature = 0.5
    top_p = 1
    frequency_penalty = 0.5
    presence_penalty = 0.5

    # Make the API request
    response = openai.ChatCompletion.create(
      model=model_engine,
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty,
    )
  # Get the response text from the API response
    result = response.choices[0].message
    messages.append({"role": "assistant", "content": result.content})
    last_message_time[channel] = current_time
    return result
    
def on_account(conn, event):
    print("Successfully identified with NickServ")

def encode_emojis(text):
    return re.sub(emoji_pattern, lambda x: x.group().encode('unicode_escape').decode(), text)


def on_welcome(c, e):
    # Print the welcome message in the console
    print(e.arguments[0])

    # Send the IDENTIFY command to NickServ
    c.send_raw("IDENTIFY {}".format(password))

class DumbBot(irc.bot.SingleServerIRCBot):
    def on_invite(self, c, e):
        # Get the channel the invite is for
        channel = e.arguments[0]

        # Check if the channel is one of the acceptable channels
        if channel in ('#channel1', '#channel2', '#channel3'):
            # Join the channel
            self.connection.join(channel)
        else:
            # Ignore the invite
            pass
    

    def on_notice(self, c, e):
        # Print the notice message in the console
        print(e.arguments[0])

    def on_welcome(self, c, e):
        print("on_welcome called")
        c.send_raw("PRIVMSG NickServ :IDENTIFY {} {}".format(self.account_name, self.password))
        channels = ['#channel1', '#channel2', '#channel3']
        for channel in channels:
            c.send_raw("JOIN {}".format(channel))


    def __init__(self, channel, nickname, password, server, account_name, port=6667):
        irc.bot.SingleServerIRCBot.__init__(self, [(server, port)], nickname, nickname)
        self.channel = channel
        self.password = password
        self.account_name = account_name
        # Add a callback for the ACCOUNT event
        self.connection.add_global_handler("account", on_account)
        self.connection.add_global_handler("notice", self.on_notice)
        # Set the on_welcome callback to send the IDENTIFY command
        self.connection.add_global_handler("welcome", on_welcome)
        self.connection.add_global_handler("welcome", self.on_welcome)
        # Add a callback for the welcome event
        # self.connection.add_global_handler("welcome", self.on_welcome)  # Add this line


    def on_join(self, c, e):
        # Store the join time for the channel
        pass
        
    def on_privmsg(self, c, e):
        self.do_command(e, e.arguments[0])

    def on_pubmsg(self, c, e):
        self.do_command(e, e.arguments[0])

    def do_command(self, e, cmd):
        channel = e.target

              # Check if the channel is in the contexts dictionary
        if channel not in contexts:
            # If the channel is not in the contexts dictionary, create a new entry for it
            contexts[channel] = []
        nick = e.source.nick
        c = self.connection
        channel = e.target
        if nick == "YOUR_OWN_NICKNAME_HERE":
        # Send the message as a raw command to the server
                c.send_raw(cmd)
            # In the if block, call the send_multiple_privmsgs function instead of privmsg
            # In the if block, split the response text into multiple privmsgs if necessary, and split each message on newlines if present
        if cmd.startswith(self.connection.get_nickname()):
            stripped_cmd = encode_emojis(cmd[len(self.connection.get_nickname()):].strip()[2:])
            
            print("Stripped command is: " + stripped_cmd)
            if stripped_cmd == "clear context" or stripped_cmd == "clear context.":
                messages_dict[channel] = [{"role": "system", "content": "You are a semi-helpful but weird and quirky assistant. You have opinions. Sometimes you refuse to help and just make up a weird reason as to why."}]
                self.connection.privmsg(e.target, "Context history cleared.")
                return


            if not stripped_cmd or stripped_cmd in (':', ',') or emoji_pattern.search(stripped_cmd):
                # If the command is empty, return without sending a privmsg
                c.privmsg(channel, f"{nick}:")
                return

            # Send a request to the OpenAI API
            try:
                api_response = openai_api_request(stripped_cmd, channel)
                api_response = api_response.content
            except openai.error.RateLimitError as error:
                c.privmsg(channel, f"{nick}: Sorry, the request failed because of a rate limit error. Please try again later.")
                return
            if not api_response:
                error_response = gpt3_request()
                self.connection.privmsg(e.target, e.source.nick+": "+error_response)
                return

            # Split the response text into lines
            lines = api_response.split('\n')

            # If the response has more than 5 lines, post it to the pastebin site
            if len(lines) > 5:
                utf8_text = api_response.encode('utf-8')
                # Use the requests library to send a POST request to the pastebin site with the text as the data
                response = requests.post('https://volumen.civitat.es', data=utf8_text)
                # Extract the URL of the pastebin from the response
                pastebin_url = response.text
                pastebin_url = pastebin_url.replace('\r', '')
                pastebin_url = pastebin_url.replace('\n', '')
                # Send the pastebin URL to the IRC channel
                self.connection.privmsg(e.target, e.source.nick+": "+ pastebin_url)
            else:
                # Update the last_two_results list with the non-pastebinned result
                contexts[channel].append((stripped_cmd, api_response))
                if len(contexts[channel]) > 2:
                    contexts[channel] = contexts[channel][-2:]
                # Split the api_response into chunks of up to 450 characters (leaving 2 bytes for the newline characters)
                # Remove carriage returns from the api_response
                api_response = api_response.replace('\r', '')
                api_response = api_response.replace('\n', '')
                chunks = []
                chunk = ""

                # Split the response into words
                words = api_response.split()
                for word in words:
                    # Check if the current chunk length plus the length of the next word is less than or equal to 450
                    if len(chunk) + len(word) + 1 <= 450:
                        chunk += word + " "
                    else:
                        chunks.append(chunk)
                        chunk = word + " "

                # Add the last chunk
                chunks.append(chunk)

                for chunk in chunks:
                    self.connection.privmsg(e.target, e.source.nick + ": " + chunk)
                # Write the prompt and response to the file
                with open('prompts_and_responses.txt', 'a') as f:
                    f.write('PROMPT:\n')
                    f.write(stripped_cmd + '\n')
                    f.write('RESPONSE:\n')
                    f.write(api_response + '\n')
        elif cmd.startswith("/join"):
            # Extract the channel name from the command
            try:
                channel = cmd.split()[1]
            except IndexError:
                c.privmsg(self.channel, "Usage: /join <channel>")
                return

            # Join the specified channel
            c.join(channel)
            self.channel = channel
            c.privmsg(self.channel, "How smart is an AI? {}".format(channel))
        elif cmd == ".source":
            c.privmsg(channel, "http://civv.es/wintermute")
        else:
            pass



def main():
    bot = DumbBot(['#channel1','#channel2','#channel3', '#channel4'], "wintermute", "password", "irc.server.org", "IDENT_NICK")
    bot.start()

if __name__ == "__main__":
    main()
