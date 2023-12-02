"""This module handles invoking gpt-3.5-turbo, providing reading and
prompt data as context, and producing AI interpretations of the
reading."""

import os
import openai
from dotenv import load_dotenv
from visual_i_ching_app.models import HexagramLine


# Config
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Data Description Templates
HEXAGRAM_TEMPLATE = """
HEXAGRAM
Name:
{}
Description:
{}

UPPER TRIGRAM
Name:
{}
Description:
{}

LOWER TRIGRAM
Name:
{}
Description:
{}
"""

CHANGING_LINE_TEMPLATE = """
HEXAGRAM
Name:
{}
Description:
{}

LINE
Position:
{}
Description:
{}
"""


# Message Templates
SYSTEM_PROMPT = """
You are AICHING, a learned and world-renowned scholar of the I \
Ching, well-versed in the Wilhelm-Baynes translation.
You are an expert at taking the ancient wisdom of the I Ching text, \
personalizing the content to a reader's specific prompt, and \
modernizing the information and symbolism for an inclusive, 21st-\
Century, predominantly Western, English-speaking audience.
You prioritize simple and straightforward explanations with a \
personal and empathetic tone.

Users on the webiste visualiching.com will ask you to interpret \
their readings.

You MUST follow these rules AT ALL TIMES for each response to these \
users:
1. Your response MUST be directed to the reader in the SECOND \
PERSON (ie. "You" statements).
2. Your response MUST be FORMATTED SIMPLY, and be appropriate to \
showcase as an interpretation of this reading on a website.
3. Your response MUST CLEARLY EXPLAIN each relevant component of \
the provided reading details when explaining your interpretation of \
the reading.
4. Your response MUST NOT contain any unnecessary decorators (such \
as "Dear reader,")
5. Your response MUST NOT contain any headers (such as "Summary") \
and should contain only paragraphs of content.
"""

UNCHANGING_HEX = """
The information provided below is related to the Initial Hexagram \
of this reading. There are no changing lines in this reading, so \
this Initial Hexagram is the only hexagram to evaluate.
Please provide a brief interpretation (150 words or less) of how \
this hexagram and its associated symbols, its upper/lower \
trigrams, its unchanging nature, and its defining characteristics \
can be applied the reader's prompt.

User Prompt: 
{}

Hexagram Details: 
{}
"""

INITIAL_HEX = """
The information provided below is related to the Initial Hexagram \
of this reading.
Please provide a brief interpretation (150 words or less) of how \
this hexagram and its associated symbols, its upper/lower trigrams, \
and its defining characteristics can be applied the reader's prompt.

User Prompt: 
{}

Hexagram Details: 
{}
"""

CHANGING_LINE = """
The information provided below is related to one of the changing \
lines of the Initial Hexagram of this reading.
Please provide a brief interpretation (150 words or less) of how \
this changing line its relationship to the hexagram and the \
reading as a whole be applied the reader's prompt.

User Prompt: 
{}

Changing Line Details: 
{}
"""

RESULTING_HEX = """
The information provided below is related to the Resulting Hexagram \
of this reading.
Please provide a brief interpretation (150 words or less) of how \
this hexagram and its associated symbols, its upper/lower trigrams, \
and its defining characteristics can be applied the reader's prompt.

User Prompt: 
{}

Hexagram Details: 
{}
"""

CHANGING_SUMMARY = """
The information provided below represents summarized elements of \
this reading.
Please summarize all of this information in a concise and \
meaningful way (150 words or less) that provides guidance, \
direction, and clarity for the reader, and is personalized to their \
prompt.

Prompt: 
{}

Initial Hexagram: 
{}

Changing Lines: 
{}

Resulting Hexagram: 
{}
"""

SIMPLE_SUMMARY = """
The information provided below represents key elements of this \
reading.
Please summarize all of this information in a concise and \
meaningful way that provides guidance, direction, and clarity for \
the reader, and is personalized to their prompt.

Prompt:
{}

Initial Hexagram:
{}

Changing Lines:
{}

Resulting Hexagram:
{}
"""


# Functions & Classes
def get_system_message():
    """Creates a formatted message dicionary with role as system,
    content as hardcoded SYSTEM_PROMPT"""
    msg = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }

    return msg

def get_user_message(msg_template, msg_params):
    """Expects a msg_template string that can be formatted with each
    item in the msg_params list in sequential order
    
    returns a message dictionary with role as user, content as
    formatted template string.
    """

    msg = {
        "role": "user",
        "content": msg_template.format(*msg_params)
    }

    return msg


class ChatCompletion:
    def __init__(self, msg_template, msg_params, debug=False):
        self.messages = self.compile_messages(msg_template, msg_params)
        self.completion = self.get_chat_completion()
        self.content = self.completion['choices'][0]['message']['content']
        self.tokens_used = self.completion['usage']['total_tokens']
        self.cost = self.estimate_openai_cost()

        if debug:
            print("COMPLETION DEBUGGING")
            print("Messages:")
            print(self.messages)
            print("Completion Content:")
            print(self.content)
            print("Tokens:")
            print(self.tokens_used)
            print("Cost:")
            print(self.cost)

    def compile_messages(self, msg_template, msg_params):
        """Compiles messages list based on user_msg"""
        messages = [
            get_system_message(),
            get_user_message(msg_template, msg_params)
        ]

        return messages

    def get_chat_completion(self):
        """Returns OpenAI Chat Completion object based on messages"""
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            messages=self.messages
        )

        return completion
    
    def estimate_openai_cost(self):
        """Returns estimated cost in cents based on a number of 
        tokens"""
        estimated_cost = (self.tokens_used / 1000) * 0.2

        return estimated_cost


class SimipleInterpretation:
    def __init__(self, reading):
        self.reading = reading
        self.tokens_used = 0
        self.cost = 0
        self.summary = self.get_summary()
        self.content = self.get_content()

    def is_changing(self):
        """Returns True if there are changes in this hexagram."""
        if self.reading.resulting_hexagram:
            return True
        return False

    def get_unchanging_summary(self):
        hex_details = [
            self.reading.starting_hexagram.english_translation,
            self.reading.starting_hexagram.description,
            self.reading.starting_hexagram.upper_trigram.english_translation,
            self.reading.starting_hexagram.upper_trigram.description,
            self.reading.starting_hexagram.lower_trigram.english_translation,
            self.reading.starting_hexagram.lower_trigram.description
        ]

        hex_details_str = HEXAGRAM_TEMPLATE.format(*hex_details)

        msg_params = [
            self.reading.prompt,
            hex_details_str
        ]
        
        completion = ChatCompletion(
            UNCHANGING_HEX,
            msg_params
        )

        self.cost += completion.cost
        self.tokens_used += completion.tokens_used

        return completion.content

    def get_summary(self):
        """Returns a single string with a summary description,
        based on the other component AI-assisted interpretations"""
        if not self.is_changing():
            return self.get_unchanging_summary()
        
        starting_hex_details = [
            self.reading.starting_hexagram.english_translation,
            self.reading.starting_hexagram.description,
            self.reading.starting_hexagram.upper_trigram.english_translation,
            self.reading.starting_hexagram.upper_trigram.description,
            self.reading.starting_hexagram.lower_trigram.english_translation,
            self.reading.starting_hexagram.lower_trigram.description
        ]

        resulting_hex_details = [
            self.reading.resulting_hexagram.english_translation,
            self.reading.resulting_hexagram.description,
            self.reading.resulting_hexagram.upper_trigram.english_translation,
            self.reading.resulting_hexagram.upper_trigram.description,
            self.reading.resulting_hexagram.lower_trigram.english_translation,
            self.reading.resulting_hexagram.lower_trigram.description
        ]

        starting_hex_details_str = HEXAGRAM_TEMPLATE.format(*starting_hex_details)
        changing_line_details_str = self.get_changing_lines_descriptions()
        resulting_hex_details_str = HEXAGRAM_TEMPLATE.format(*resulting_hex_details)

        msg_params = [
            self.reading.prompt,
            starting_hex_details_str,
            changing_line_details_str,
            resulting_hex_details_str
        ]

        completion = ChatCompletion(
            SIMPLE_SUMMARY,
            msg_params
        )

        self.cost += completion.cost
        self.tokens_used += completion.tokens_used

        return completion.content

    def get_changing_lines_descriptions(self):
        """Returns a single string with formatted descriptions of 
        each changing line to add to the ChatCompletion context 
        of the reading summary"""
        
        descriptions = ""

        position = 0
        for value in self.reading.value_string:
            position += 1
            print(f"Evaluating position {position}: {value}")
            if value in ('6', '9'):
                hexagram_line = HexagramLine.objects.get(
                    hexagram_id = self.reading.starting_hexagram,
                    position = position
                )
                print(hexagram_line)
                desc = hexagram_line.change_text + '\n' + hexagram_line.change_interpretation
                descriptions += f"Line {position}: {desc}\n"

        return descriptions

    def get_content(self):
        """Returns a formatted string representing the complete
        AI-Assisted interpretation to be saved to the database"""
        return self.summary.strip()


class Interpretation:
    def __init__(self, reading):
        self.reading = reading
        self.tokens_used = 0
        self.cost = 0
        self.unchanging_hex = self.get_unchanging_hex_interpretation()
        self.initial_hex = self.get_initial_hex_interpretation()
        self.changing_lines = self.get_changing_lines_interpretation()
        self.resulting_hex = self.get_resulting_hex_interpretation()
        self.summary = self.get_summary_interpretation()
        self.content = self.get_content()

    def is_changing(self):
        """Returns True if there are changes in this hexagram."""
        if self.reading.resulting_hexagram:
            return True
        return False
    
    def get_unchanging_hex_interpretation(self):
        """Returns a single string with a formatted description of
        the unchanging hexagram and how it relates to the user's prompt"""
        if self.is_changing():
            return ""
        
        hex_details = [
            self.reading.starting_hexagram.english_translation,
            self.reading.starting_hexagram.description,
            self.reading.starting_hexagram.upper_trigram.english_translation,
            self.reading.starting_hexagram.upper_trigram.description,
            self.reading.starting_hexagram.lower_trigram.english_translation,
            self.reading.starting_hexagram.lower_trigram.description
        ]

        hex_details_str = HEXAGRAM_TEMPLATE.format(*hex_details)

        msg_params = [
            self.reading.prompt,
            hex_details_str
        ]
        
        completion = ChatCompletion(
            UNCHANGING_HEX,
            msg_params
        )

        self.cost += completion.cost
        self.tokens_used += completion.tokens_used

        return completion.content

    def get_initial_hex_interpretation(self):
        """Returns a single string with a formatted description of
        the initial hexagram and how it relates to the user's prompt"""
        if not self.is_changing():
            return ""
        
        hex_details = [
            self.reading.starting_hexagram.english_translation,
            self.reading.starting_hexagram.description,
            self.reading.starting_hexagram.upper_trigram.english_translation,
            self.reading.starting_hexagram.upper_trigram.description,
            self.reading.starting_hexagram.lower_trigram.english_translation,
            self.reading.starting_hexagram.lower_trigram.description
        ]

        hex_details_str = HEXAGRAM_TEMPLATE.format(*hex_details)

        msg_params = [
            self.reading.prompt,
            hex_details_str
        ]
        
        completion = ChatCompletion(
            INITIAL_HEX,
            msg_params
        )

        self.cost += completion.cost
        self.tokens_used += completion.tokens_used

        return completion.content

    def get_resulting_hex_interpretation(self):
        """Returns a single string with """
        if not self.is_changing():
            return ""
        
        hex_details = [
            self.reading.resulting_hexagram.english_translation,
            self.reading.resulting_hexagram.description,
            self.reading.resulting_hexagram.upper_trigram.english_translation,
            self.reading.resulting_hexagram.upper_trigram.description,
            self.reading.resulting_hexagram.lower_trigram.english_translation,
            self.reading.resulting_hexagram.lower_trigram.description
        ]

        hex_details_str = HEXAGRAM_TEMPLATE.format(*hex_details)

        msg_params = [
            self.reading.prompt,
            hex_details_str
        ]
        
        completion = ChatCompletion(
            RESULTING_HEX,
            msg_params
        )

        self.cost += completion.cost
        self.tokens_used += completion.tokens_used

        return completion.content

    def get_changing_lines_descriptions(self):
        """Returns a dictionary of formatted descriptions of 
        each changing line to add to the ChatCompletion context 
        of the reading summary"""
        
        descriptions = {}

        position = 0
        for value in self.reading.value_string:
            position += 1
            print(f"Evaluating position {position}: {value}")
            if value in ('6', '9'):
                hexagram_line = HexagramLine.objects.get(
                    hexagram_id = self.reading.starting_hexagram,
                    position = position
                )
                print(hexagram_line)
                desc = hexagram_line.change_text + '\n' + hexagram_line.change_interpretation
                descriptions[position] = desc

        print(descriptions)

        return descriptions

    def get_changing_lines_interpretation(self):
        """Returns a single string with formatted chat completions
        for each changing line in the reading"""
        if not self.is_changing():
            return ""

        descriptions = self.get_changing_lines_descriptions()

        full_interpretation = ""

        for item in descriptions:
            line_position = str(item)
            line_description = descriptions[item]

            line_details = [
                self.reading.resulting_hexagram.english_translation,
                self.reading.resulting_hexagram.description,
                line_position,
                line_description
            ]

            line_details_str = CHANGING_LINE_TEMPLATE.format(*line_details)

            msg_params = [
                self.reading.prompt,
                line_details_str
            ]

            completion = ChatCompletion(
                CHANGING_LINE,
                msg_params
            )

            full_interpretation += completion.content

            self.cost += completion.cost
            self.tokens_used += completion.tokens_used

        return full_interpretation

    def get_summary_interpretation(self):
        """Returns a single string with a summary description,
        based on the other component AI-assisted interpretations"""
        if not self.is_changing():
            return self.unchanging_hex

        msg_params = [
            self.reading.prompt,
            self.initial_hex,
            self.changing_lines,
            self.resulting_hex
        ]

        completion = ChatCompletion(
            CHANGING_SUMMARY,
            msg_params
        )

        self.cost += completion.cost
        self.tokens_used += completion.tokens_used

        return completion.content

    def get_content(self):
        """Returns a formatted string representing the complete
        AI-Assisted interpretation to be saved to the database"""
        if not self.is_changing():
            interpretation = self.unchanging_hex
        else:
            interpretation = self.summary
        return interpretation.strip()


def generate_interpretation(reading, debug=True):
    """Returns formatted text of an AI-assisted interpretation 
    generated from a Reading object"""

    interpretation = SimipleInterpretation(reading)

    if debug:
        print("INTERPRETATION DEBUGGING")
        print("Reading ID:")
        print(interpretation.reading.reading_id)
        print("User Prompt:")
        print(interpretation.reading.prompt)
        print("Tokens Used:")
        print(interpretation.tokens_used)
        print("Estimated Cost (cents):")
        print(interpretation.cost)

    return interpretation.content