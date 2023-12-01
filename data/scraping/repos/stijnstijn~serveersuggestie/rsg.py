from pathlib import Path
from random import choice
from markov import generate

import sys
import requests
import sys
import re

allow_robots = True
forbidden = "^ \U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BFa-zA-Z0-9|\[\]()!@=/?*#$%\\^_&*.,;':\"-+<>-"
forbidden = "/"
pattern = re.sub(r"[" + forbidden + "]", "", " ".join(Path("rsg.temp").read_text().strip().split(" ")[1:]), flags=re.UNICODE)
pattern = " ".join(Path("rsg.temp").read_text().strip().split(" ")[1:])

if not pattern or re.match(r"^[0-9]+$", pattern):
    allow_robots = True
    templates = [item for item in Path("rsg.templates").read_text().split("\n") if item.strip()]
    if re.match(r"^[0-9]+$", pattern) and int(pattern) <= len(templates) and int(pattern) > 0:
        pattern = templates[int(pattern) - 1]
    else:
        pattern = choice(templates)

if pattern != "zelfde":
    Path("rsg.pattern").write_text(pattern)

replacements = []
pattern = Path("rsg.pattern").read_text()

channel = sys.argv[1] if len(sys.argv) > 1 else ""
pattern = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else pattern
result = []

userfile = Path("rsg.nicks")
if userfile.exists():
    with userfile.open() as userstream:
        users = [user.strip() for user in userstream.read().split("\n") if user.strip()]
else:
    users = ["gonzobot"]

def get_word_from_bank(file=None, pattern=None, default="", bank=None):
    if bank:
        options = bank
    elif Path(file).exists():
        options = [option.strip() for option in Path(file).open().readlines()]
    else:
        return default

    if pattern:
        options = [option for option in options if re.match(pattern, option, flags=re.IGNORECASE)]
        return choice(options) if options else default
    else:
        return choice(options)


def parse(buffer):
    global channel, users

    replacement = ""
    capitalise = False
    lowercase = False
    all_capitals = False
    if not buffer:
        return ""

    regex = re.search(r"==(.+)$", buffer)
    pattern = None
    if regex:
       buffer = buffer[:len(buffer)-len(regex.group(0))]
       pattern = "^" + re.escape(regex.group(1)).replace("\\*", "*").replace("\\?", "?").replace("?", ".").replace("*", ".*")+ "$"

    if buffer[-1] == "^":
        capitalise = True
        buffer = buffer[:-1]
    elif buffer[-1] == "_":
        lowercase = True
        buffer = buffer[:-1]

    if buffer == buffer.upper() and buffer.upper() != buffer.lower():
        all_capitals = True

    buffer = buffer.lower()
    if "|" in buffer:
        buffer = choice(buffer.split("|"))

    for character in buffer:
        if character == "u":
            replacement += get_word_from_bank(bank=users, pattern=pattern)
        elif character in ("🚖", "🚗", "🚘", "🚙"):
            replacement += get_word_from_bank(file="banks/automerken.txt", pattern=pattern)
        elif character == "🍕":
            replacement += "Pizza " + get_word_from_bank(file="banks/pizza.txt", pattern=pattern)
        elif character in ("🚲"):
            replacement += get_word_from_bank(file="banks/fietsmerken.txt", pattern=pattern)
        elif character in ("🚂", "🚄", "🚅", "🚞", "🚆", "🚇"):
            replacement += get_word_from_bank(file="banks/treinen.txt", pattern=pattern)
        elif character in ("✈"):
            replacement += get_word_from_bank(file="banks/vliegtuigen.txt", pattern=pattern)
            break
        elif character in ("🤖"):
            if allow_robots:
                replacement += "%%OPENAI_INSERT%%"
            break
        elif character in ("🪐"):
            replacement += get_word_from_bank(file="banks/planeten.txt", pattern=pattern)
        elif character == "v" and len(buffer) > 1 and buffer[1] == "n":
            which = choice(("m", "v", "x"))
            if len(buffer) > 2 and buffer[2] == ":":
                bits = buffer.split(":")
                if bits[1] in ("v", "m", "x"):
                    which = bits[1]
            replacement += get_word_from_bank(file=f"banks/voornamen-{which}.txt", pattern=pattern)
            break
        elif character == "a" and len(buffer) > 1 and buffer[1] == "n":
            replacement += get_word_from_bank(file="banks/achternamen.txt", pattern=pattern)
            break
        elif character == "v":
            replacement += get_word_from_bank(bank=["a","e","i","o","u","y"], pattern=pattern)
            break
        elif character == "c":
            replacement += get_word_from_bank(bank=["b","c","d","f","g","h","j","k","l","m","n","p","q","r","s","t","v","w","x","z"], pattern=pattern)
        elif character == "n":
            plural = len(buffer) > 1 and buffer[1] == "s"
            noun = get_word_from_bank(file="banks/zelfstnw.txt", pattern=pattern).split("|")
            if len(noun) == 1 or not plural:
                 replacement += noun[0]
            elif len(noun) == 2 and plural:
                 replacement += noun[1]
        elif character == "e":
            replacement += get_word_from_bank(file="banks/bedrijven.txt", pattern=pattern)
        elif character == "g":
            replacement += get_word_from_bank(file="banks/genres.txt", pattern=pattern)
        elif character == "i":
            which = choice(("lit", "shit"))
            if len(buffer) > 1 and buffer[1] == ":":
                bits = buffer.split(":")
                bits[1] = bits[1].replace("list", "")
                if bits[1] in ("lit", "shit"):
                    which = bits[1]
            try:
                replacement += requests.get("https://api.tnl.social/list/%s" % which,
                                            timeout=5).text
            except (requests.RequestException, ConnectionRefusedError) as e:
                replacement += "stijn" if which == "lit" else "poep"
            break
        elif character == "a":
            replacement += get_word_from_bank(file="banks/bijvnw.txt", pattern=pattern)
        elif character == "f":
            replacement += get_word_from_bank(file="banks/apps.txt", pattern=pattern)
        elif character == "s":
            replacement += get_word_from_bank(file="banks/scheldwoorden.txt", pattern=pattern)
        elif character == "p":
            replacement += get_word_from_bank(file="banks/beroemdheden.txt", pattern=pattern)
        elif character == "b":
            replacement += get_word_from_bank(file="banks/partijen.txt", pattern=pattern)
        elif character in ("l", "🏙", "🌃", "🌆", "🌇"):
            replacement += get_word_from_bank(file="banks/plaatsen.txt", pattern=pattern)
        elif character in ("🏳", "🌍", "🌎", "🌏", "🗺"):
            replacement += get_word_from_bank(file="banks/landen.txt", pattern=pattern)
        elif character == "<":
            try:
                index = max(1, int(buffer[1:]))
            except ValueError:
                replacement = character + buffer
                break
            if index <= len(replacements):
                replacement = replacements[index - 1]
                break
        elif character == "#":
            nmin = None
            if len(buffer) > 1 and buffer[1] == ":":
                bits = buffer.split(":")[1].split("-")
                if len(bits) == 2:
                    try:
                        nmin = int(bits[0])
                        nmax = int(bits[1])
                        if nmin >= nmax:
                            raise ValueError()
                    except ValueError:
                        nmin = 0
                        nmax = 100
            if not nmin:
                nmin = 0
                nmax = 100
            replacement += str(choice(range(nmin, nmax)))
            break
        elif character == "m" and buffer[1] == ":":
            corpus = buffer.split(":")[1]
            if corpus in ("speld", "surpator", "linus"):
                try:
                    replacement += requests.get("https://api.tnl.social/markov/%s" % corpus, timeout=5).text
                except (requests.RequestException, ConnectionRefusedError) as e:
                    replacement += "stijn wint markovcompetitie 2020 en nu krijgt hij elke week basilicumplantje dat direct sterft " if corpus == "speld" else "Actie tegen hondelulzonneklepwafelijzers in het OV"
                break

            corpus += ".txt"
            sentence = generate(corpus, pattern=pattern, max_attempts=250)
            if sentence is not None:
                 replacement = sentence
                 break
            else:
                 replacement += character
        else:
            replacement += character

    if lowercase:
        replacement = replacement.lower()

    if capitalise:
        replacement = replacement.title()

    if all_capitals:
        replacement = replacement.upper()

    replacements.append(replacement)
    return replacement

cursor = 0
while True:
    if cursor > len(pattern) - 1:
        break

    character = pattern[cursor]

    if character == "[":
        buffer = ""
        buffer_closed = False
        buffer_cursor = cursor + 1

        while buffer_cursor < len(pattern):
            buffer_character = pattern[buffer_cursor]

            if buffer_character == "]":
                buffer_closed = True
                break

            buffer += buffer_character
            buffer_cursor += 1

        if not buffer_closed:
            print("Unclosed parentheses")
            exit()

        cursor += len(buffer) + 1
        replacement = parse(buffer)
        result.append(replacement)

    else:
        result.append(character)

    cursor += 1

result = "".join(result)
if "%%OPENAI_INSERT%%" in result:
    import openai
    openai.api_key = Path("openapi.key").read_text().strip()
    bits = result.split("%%OPENAI_INSERT%%")
    prompt = bits[0]
    suffix = "🤖".join(bits[1:])

    response = openai.Completion.create(
        engine="text-davinci-002",
        temperature=0.75,
        prompt=prompt,
        suffix=suffix,
        max_tokens=128,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0
    )

    response = response["choices"][0]["text"].strip().split("\n")[0]
    result = prompt + response + suffix

print((" " + result)[0:500])
