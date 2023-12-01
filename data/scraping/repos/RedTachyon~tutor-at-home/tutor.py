from enum import Enum
import xml.etree.ElementTree as ET
from anthropic import Anthropic

from prompts import *

MAX_TOKENS = 500


class State(Enum):
    START = 0
    CORRECT = 1
    FIRSTHINT = 2
    HINT_OR_SOL_OR_REVISE = 3
    HINT_LOOP = 4
    SOLUTION_CLEAR = 5
    NEW_ATTEMPT = 6
    # EXPLANATION = 7
    # EXPLANATION_DONE = 8

    END = 10


def get_tag_value(text: str, tag: str) -> str:
    try:
        # Wrap input text in <root> tags
        wrapped_text = f"<root>{text}</root>"
        # Parse the XML
        root = ET.fromstring(wrapped_text)
        # Find the tag
        tag_element = root.find(tag)
        # Return the text if tag is found, else return None
        return tag_element.text.strip() if tag_element is not None else None
    except ET.ParseError:
        # Handle the case where the text cannot be parsed
        return ""


class Tutor:
    def __init__(self, anthropic: Anthropic, question: str, solution: str, temperature: float = 0.):
        # self.anthropic = Anthropic(api_key=ANTHROPIC_KEY)
        self.anthropic = anthropic
        self.question = question
        self.solution = solution
        self.temperature = temperature

        self.context = ""
        self.state = State.START

        self.memory = {}

    def run(self, query: str):
        if self.state == State.START:
            reply = self.ask_in_context(START_PROMPT, self.question, self.solution, query)
            correct = get_tag_value(reply, "correct") == "Y"
            if correct:
                reply = self.ask_in_context(CORRECT_PROMPT)
                congrats = get_tag_value(reply, "congrats")
                res = congrats
                self.state = State.END
            else:
                reply = self.ask_in_context(WRONG_PROMPT)
                hint = get_tag_value(reply, "hint")
                question = get_tag_value(reply, "question")
                res = hint + "\n" + question
                self.memory["question"] = res
                self.state = State.HINT_OR_SOL_OR_REVISE
        elif self.state == State.HINT_OR_SOL_OR_REVISE or self.state == State.HINT_LOOP:
            reply = self.ask_separately(HINT_OR_SOL_OR_REVISE_PROMPT, self.memory["question"], query)
            hrs = get_tag_value(reply, "choice")
            # res = reply
            if hrs == "hint":
                reply = self.ask_in_context(HINT_PROMPT, query)
                hint = get_tag_value(reply, "hint")
                question = get_tag_value(reply, "question")
                res = hint + "\n" + question
                self.memory["question"] = res
                self.state = State.HINT_LOOP
            elif hrs == "revise":
                reply = self.ask_in_context(REVISION_PROMPT, query)
                encourage = get_tag_value(reply, "encourage")
                res = encourage
                self.state = State.NEW_ATTEMPT
            elif hrs == "solution":
                reply = self.ask_in_context(SOLUTION_PROMPT, query)
                solution = get_tag_value(reply, "solution")
                question = get_tag_value(reply, "question")
                res = solution + "\n" + question
                self.memory["question"] = res
                self.state = State.SOLUTION_CLEAR
            elif hrs == "unclear":
                reply = self.ask_in_context(UNCLEAR_CHOICE_PROMPT, self.question, query)
                response = get_tag_value(reply, "response")
                res = response
            elif hrs == "attempted":
                reply = self.ask_in_context(NEW_ATTEMPT_PROMPT, self.question, self.solution, query)
                correct = get_tag_value(reply, "correct") == "Y"
                if correct:
                    reply = self.ask_in_context(CORRECT_PROMPT)
                    congrats = get_tag_value(reply, "congrats")
                    res = congrats
                    self.state = State.END
                else:
                    reply = self.ask_in_context(WRONG_AGAIN_PROMPT)
                    hint = get_tag_value(reply, "hint")
                    question = get_tag_value(reply, "question")
                    res = hint + "\n" + question
                    self.memory["question"] = res
                    self.state = State.HINT_OR_SOL_OR_REVISE
            else:
                raise ValueError("Claude's got problems")

        elif self.state == State.SOLUTION_CLEAR:
            reply = self.ask_separately(SOLUTION_CLEAR_PROMPT, self.memory["question"], query)
            choice = get_tag_value(reply, "choice")
            if choice == "explain":
                reply = self.ask_in_context(EXPLANATION_PROMPT, query)
                explanation = get_tag_value(reply, "explanation")
                question = get_tag_value(reply, "question")
                res = explanation + "\n" + question
                self.memory["question"] = res
                self.state = State.SOLUTION_CLEAR
            elif choice == "solve":
                reply = self.ask_in_context(REVISION_PROMPT, query)
                encourage = get_tag_value(reply, "encourage")
                res = encourage
                self.state = State.NEW_ATTEMPT
            elif choice == "unclear":
                reply = self.ask_in_context(UNCLEAR_CHOICE_PROMPT, self.question, query)
                response = get_tag_value(reply, "response")
                res = response
            elif choice == "attempted":
                reply = self.ask_in_context(NEW_ATTEMPT_PROMPT, self.question, self.solution, query)
                correct = get_tag_value(reply, "correct") == "Y"
                if correct:
                    reply = self.ask_in_context(CORRECT_PROMPT)
                    congrats = get_tag_value(reply, "congrats")
                    res = congrats
                    self.state = State.END
                else:
                    reply = self.ask_in_context(WRONG_AGAIN_PROMPT)
                    hint = get_tag_value(reply, "hint")
                    question = get_tag_value(reply, "question")
                    res = hint + "\n" + question
                    self.memory["question"] = res
                    self.state = State.HINT_OR_SOL_OR_REVISE
            else:
                raise ValueError("Claude's got problems")
        elif self.state == State.NEW_ATTEMPT:
            reply = self.ask_in_context(NEW_ATTEMPT_PROMPT, self.question, self.solution, query)
            correct = get_tag_value(reply, "correct") == "Y"
            if correct:
                reply = self.ask_in_context(CORRECT_PROMPT)
                congrats = get_tag_value(reply, "congrats")
                res = congrats
                self.state = State.END
            else:
                reply = self.ask_in_context(WRONG_AGAIN_PROMPT)
                hint = get_tag_value(reply, "hint")
                question = get_tag_value(reply, "question")
                res = hint + "\n" + question
                self.memory["question"] = res
                self.state = State.HINT_OR_SOL_OR_REVISE
        elif self.state == State.END:
            res = ""
        else:
            res = ""
            self.state = State.END

        return res

    def naive_loop(self):
        print("Welcome to Claude, the math tutor! Try solving the problem!")
        while True:
            inp = input()
            if inp.startswith("$"):
                inp = globals()[inp[1:]]
            res = self.run(inp)
            print(res)
            if self.state == State.END:
                break

    def ask_in_context(self, prompt: str, *args):
        # print(f"LOG: {prompt}, {args}")
        prompt = self.context + "\n\n" + (prompt % args)
        completion = self.anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=MAX_TOKENS,
            prompt=prompt,
            temperature=self.temperature
        )

        result = completion.completion

        self.context = prompt + result

        return result

    def ask_separately(self, prompt: str, *args):
        prompt = prompt % args

        completion = self.anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=MAX_TOKENS,
            prompt=prompt,
            temperature=self.temperature
        )
        # print(f"LOG: {prompt}, {args}")
        # print(f"RESPONSE LOG: {completion.completion}")

        return completion.completion
