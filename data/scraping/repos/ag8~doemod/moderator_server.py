import random
import select
import socket
import subprocess
import sys
import threading
import time

import requests
from openai import OpenAI
# from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# anthropic = Anthropic(
#     api_key="",
# )

from audio_grabber import record_audio, play_audio, analyze_audio

# Run `export OPENAI_API_KEY='yourkey'`
client = OpenAI()

def check_answer(voice_input, correct_answer, question_text, question_type):
    if question_type == "Multiple Choice":
        if len(voice_input) < 10 and voice_input[0].lower() in ['w', 'x', 'y', 'z']:  # if it's just one character, we can compare manually
            return correct_answer[0].lower() == voice_input[0].lower()
    elif correct_answer[0].lower() == voice_input[0].lower():
            return True

    say(f"You said, {voice_input}.")
    print("Calling GPT...")

    if question_type == "Multiple Choice":
        prompt = f"You are evaluating an answer for Science Bowl. The question was: ```\n{question_text}\n```. The correct answer is `{correct_answer}`. According to voice transcription software, the student said `{voice_input}`. (It sometimes makes mistakes, so 'why' might be a mistransliteration of 'Y'). Should this answer be counted? Saying just the letter of the correct choice is considered correct and should be counted. If the student gave an answer in words, then the student must have given the correct answer, word for word, for the answer to be counted. Respond only YES or NO. Say YES if the answer should be accepted, and NO if the answer should not be accepted. Say only YES or NO, and nothing else."
    else:
        prompt = f"You are evaluating an answer for Science Bowl. The question was: ```\n{question_text}\n```. The correct answer is `{correct_answer}`. According to voice transcription software, the student said `{voice_input}`. Should this answer be counted? Is it essentially the correct answer, or scientifically very close? Respond only YES or NO. Say YES if the answer should be accepted, and NO if the answer should not be accepted. Say only YES or NO, and nothing else."

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1,  # YES or NO are each a single token
        temperature=0
    )

    result = response.choices[0].message.content

    # completion = anthropic.completions.create(
    #     model="claude-2",
    #     max_tokens_to_sample=300,
    #     prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    # )

    # result = completion.completion.strip()

    print(result)

    if result == "YES":
        return True
    elif result == "NO":
        return False
    else:
        print("Not sure if this answer is correct")
        print(f"Result is {result}")
        return random.random() < 0.3


def say(text, interruptible=False):
    if interruptible:
        return subprocess.Popen(["say", text])
    else:
        subprocess.run(["say", text])


class ModeratorServer:
    def __init__(self, host="0.0.0.0", port=12348):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen()
        self.player_names = {}
        self.buzz_event = threading.Event()
        self.question_number = 0
        self.buzz_pause = False
        self.current_buzzer = None
        self.scores = {}
        self.buzzed_this_question = []
        print(f"Server started on {host}:{port}. Waiting for players...")

    def get_question(self):
        url = "https://scibowldb.com/api/questions/random"

        # Make a GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            return response.json()
        else:
            # Handle errors (e.g., network issues, server errors)
            return f"Error: Unable to fetch question (Status code: {response.status_code})"

    def question(self, current_question, type="tossup", read=True, player_bonus=None):
        assert type == "tossup" or type == "bonus"
        question_text = f"Your {type} number {self.question_number} is in {current_question.get('question', '').get('category', '')}; {current_question.get('question', '').get(f'{type}_format', '')}. "
        question_text += f"{current_question.get('question').get(f'{type}_question', '')}"

        if current_question.get('question').get(f'{type}_format', '') == "Multiple Choice":
            question_text = (question_text
                             .replace("\nW)", ". [[slnc 500]] W [[slnc 250]]")
                             .replace("\nX)", ". [[slnc 500]] X [[slnc 250]]")
                             .replace("\nY)", ". [[slnc 500]] Y [[slnc 250]]")
                             .replace("\nZ)", ". [[slnc 500]] Z [[slnc 250]]"))

        if not read:
            question_text = ""

        saying_process = say(question_text, interruptible=True)

        while True:
            if self.buzz_pause:
                # Check that this person hasn't already buzzed
                if self.current_buzzer in self.buzzed_this_question:
                    saying_process.kill()
                    say(f"Bruh, you've already buzzed on this question? The fuck? Re-reading for the other player{'s' * (len(self.player_names) > 2)}.")

                    self.buzz_pause = False
                    self.current_buzzer = None

                    self.question(current_question, "tossup")

                self.buzzed_this_question.append(self.current_buzzer)

                correct_buzz = False

                # Check if the process was still running
                still_running = saying_process.poll() is None

                # Kill the process
                saying_process.kill()

                # Recognize the player
                if still_running:
                    say("Interrupt!" + self.current_buzzer + ".")
                else:
                    say(self.current_buzzer + ".")

                # Figure out how many seconds to record for based on the answer length
                # The formula is: a minimum of 3 seconds, up to 1 second per word (could be unlimited)
                # If it's multiple choice, only record for two seconds, since it's just the letter
                # However if it's an interrupt, record for the full time
                words_in_answer = current_question.get('question', '').get(f'{type}_answer', '').count(" ") + 1
                is_tossup = type == "tossup"
                is_interrupt = still_running
                record_seconds = 2 if is_tossup and not is_interrupt else max(3, words_in_answer)

                print(record_seconds)

                record_audio(f"/Users/dyusha/fun_things/doemod/recordings/answer_{self.question_number}.wav", record_seconds)
                # play_audio(f"/Users/dyusha/fun_things/doemod/recordings/answer_{self.question_number}.wav")
                voice_input = analyze_audio(f"/Users/dyusha/fun_things/doemod/recordings/answer_{self.question_number}.wav")

                # Get the correct answer
                correct_answer = current_question.get('question', '').get(f'{type}_answer', '')

                if check_answer(voice_input, correct_answer,
                                current_question.get('question', '').get(f'{type}_question', ''),
                                current_question.get('question', '').get(f'{type}_format', '')):
                    say(f"{correct_answer} is correct!")

                    self.current_buzzer = player_bonus if player_bonus is not None else self.current_buzzer
                    self.scores[self.current_buzzer] += 4 if type == "tossup" else 10
                    correct_buzz = True

                    self.buzz_pause = False
                    player = self.current_buzzer
                    self.current_buzzer = None
                    self.buzzed_this_question = []

                    if type == 'tossup':
                        self.question(current_question, 'bonus', player_bonus=player)

                else:
                    if len(self.buzzed_this_question) < 2 and type == 'tossup':
                        say("Incorrect.")
                    else:
                        say(f"Incorrect; the correct answer was {correct_answer}. Moving on.")

                    if still_running and type == 'tossup':  # interrupt; lose points
                        self.scores[self.current_buzzer] -= 4

                self.buzz_pause = False
                self.current_buzzer = None

                time.sleep(0.5)

                print(self.scores)

                if correct_buzz:
                    if type == "tossup":
                        self.question(current_question, "bonus")
                    else:
                        self.next_question()
                else:  # incorrect buzz; re-read if there's been less than 2 buzzes
                    if type == "bonus":
                        self.next_question()
                    else:
                        if len(self.buzzed_this_question) >= 2:
                            self.next_question()
                        else:
                            if still_running:
                                say(f"I'll re-read for the other player{'s' * (len(self.player_names) > 2)}")

                                self.question(current_question, "tossup")
                            else:
                                self.question(current_question, "tossup", read=False)

    def next_question(self):
        if self.question_number == 16:
            # Sort the dictionary by scores in increasing order
            sorted_scores = sorted(self.scores.items(), key=lambda x: x[1])

            # Construct the announcement
            announcement = "And that's the round! [[slnc 500]] "
            for name, score in sorted_scores[:-1]:  # Go through all but the highest score
                announcement += f"{name} with {score} points. [[slnc 500]] "

            # Add a longer pause before announcing the first place
            announcement += "[[slnc 1000]] And in first place, "
            announcement += f"{sorted_scores[-1][0]} with {sorted_scores[-1][1]} points! Congratulations to the winner!"

            final_statement = announcement

            say(final_statement)

            sys.exit(0)

        self.buzzed_this_question = []

        self.question_number += 1
        current_question = self.get_question()

        self.question(current_question, "tossup")

    def client_thread(self, conn, addr):
        conn.send("Enter your name: ".encode())
        name = conn.recv(1024).decode().strip()
        self.player_names[conn] = name
        print(f"{name} from {addr} has joined the game.")

        while True:
            try:
                ready_to_read, _, _ = select.select([conn], [], [], 0.1)
                if ready_to_read:
                    _ = conn.recv(1024)  # Client buzzed
                    print(f"{self.player_names[conn]} buzzed!")
                    self.buzz_pause = True
                    self.current_buzzer = self.player_names[conn]
            except Exception as e:
                print(f"Error with client {addr}: {e}")
                break

    def run(self):
        threading.Thread(target=self.accept_clients).start()

        command = input("Type 'start' to begin counting: ").strip().lower()
        if command == 'start':
            print("Starting...")

            # Initialize all scores to zero
            for player_name in self.player_names.values():
                self.scores[player_name] = 0

            threading.Thread(target=self.next_question).start()

    def accept_clients(self):
        try:
            while True:
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self.client_thread, args=(conn, addr)).start()
        except Exception as e:
            print(f"Error accepting clients: {e}")


if __name__ == "__main__":
    server = ModeratorServer()
    server.run()
