"""
#playground.py
"""

"""
Q&A
Answer questions based on existing knowledge.

@Tag:Answers, @Tag:Generation, @Tag:Conversation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ: Where is the Valley of Kings?\nA:",
  temperature=0,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)

"""
Grammar correction
Corrects sentences into standard English.

@Tag:Transformation, @Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Correct this to standard English:\n\nShe no went to the market.",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Summarize for a 2nd grader
Translates difficult text into simpler concepts.

@Tag:Transformation, @Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Summarize this for a second-grade student:\n\nJupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.",
  temperature=0.7,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Natural language to OpenAI API
Create code to call to the OpenAI API using a natural language instruction.

@Tag:Code, @Tag:Transformation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="\"\"\"\nUtil exposes the following:\nutil.openai() -> authenticates & returns the openai module, which has the following functions:\nopenai.Completion.create(\n    prompt=\"<my prompt>\", # The prompt to start completing from\n    max_tokens=123, # The max number of tokens to generate\n    temperature=1.0 # A measure of randomness\n    echo=True, # Whether to return the prompt in addition to the generated completion\n)\n\"\"\"\nimport util\n\"\"\"\nCreate an OpenAI completion starting from the prompt \"Once upon an AI\", no more than 5 tokens. Does not include the prompt.\n\"\"\"\n",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\"\"\""]
)

"""
Text to command
Translate text into programmatic commands.

@Tag:Transformation, @Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Convert this text to a programmatic command:\n\nExample: Ask Constance if we need some bread\nOutput: send-msg `find constance` Do we need some bread?\n\nContact the ski store and figure out if I can get my skis fixed before I leave on Thursday",
  temperature=0,
  max_tokens=100,
  top_p=1.0,
  frequency_penalty=0.2,
  presence_penalty=0.0,
  stop=["\n"]
)

"""
English to other languages
Translates English text into French, Spanish and Japanese.

@Tag:Transformation, @Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="### Postgres SQL tables, with their properties:\n#\n# Employee(id, name, department_id)\n# Department(id, name, address)\n# Salary_Payments(id, employee_id, amount, date)\n#\n### A query to list the names of the departments which employed more than 10 employees in the last 3 months\nSELECT",
  temperature=0,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["#", ";"]
)

"""
Natural language to Stripe API
Create code to call the Stripe API using natural language.

@Tag:Code, @Tag:Transformation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="\"\"\"\nUtil exposes the following:\n\nutil.stripe() -> authenticates & returns the stripe module; usable as stripe.Charge.create etc\n\"\"\"\nimport util\n\"\"\"\nCreate a Stripe token using the users credit card: 5555-4444-3333-2222, expiration date 12 / 28, cvc 521\n\"\"\"",
  temperature=0,
  max_tokens=100,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\"\"\""]
)

"""
SQL translate
Translate natural language to SQL queries.

@Tag:Code, @Tag:Transformation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="### Postgres SQL tables, with their properties:\n#\n# Employee(id, name, department_id)\n# Department(id, name, address)\n# Salary_Payments(id, employee_id, amount, date)\n#\n### A query to list the names of the departments which employed more than 10 employees in the last 3 months\nSELECT",
  temperature=0,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["#", ";"]
)


"""
Parse unstructured data
Create tables from long form text

@Tag:Transformation, @Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="A table summarizing the fruits from Goocrux:\n\nThere are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy. There are also loheckles, which are a grayish blue fruit and are very tart, a little bit like a lemon. Pounits are a bright green color and are more savory than sweet. There are also plenty of loopnovas which are a neon pink flavor and taste like cotton candy. Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them.\n\n| Fruit | Color | Flavor |",
  temperature=0,
  max_tokens=100,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)


"""
Classification
Classify items into categories via example.

@Tag:Classifiers
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="The following is a list of companies and the categories they fall into:\n\nApple, Facebook, Fedex\n\nApple\nCategory:",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)


"""
Python to natural language
Explain a piece of Python code in human understandable language.

@Tag:Code, @Tag:Translation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="# Python 3 \ndef remove_common_prefix(x, prefix, ws_prefix): \n    x[\"completion\"] = x[\"completion\"].str[len(prefix) :] \n    if ws_prefix: \n        # keep the single whitespace as prefix \n        x[\"completion\"] = \" \" + x[\"completion\"] \nreturn x \n\n# Explanation of what the code does\n\n#",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["#"]
)

"""
Movie to Emoji
Convert movie titles into emoji.

@Tag:Transformation, @Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Convert movie titles into emoji.\n\nBack to the Future: 👨👴🚗🕒 \nBatman: 🤵🦇 \nTransformers: 🚗🤖 \nStar Wars:",
  temperature=0.8,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)

"""
Calculate Time Complexity
Find the time complexity of a function.

@Tag:Code, @Tag:Transformation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="def foo(n, k):\naccum = 0\nfor i in range(n):\n    for l in range(k):\n        accum += i\nreturn accum\n\"\"\"\nThe time complexity of this function is",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)

"""
Translate programming languages
Translate from one programming language to another

@Tag:Code, @Tag:Translation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="##### Translate this function  from Python into Haskell\n### Python\n    \n    def predict_proba(X: Iterable[str]):\n        return np.array([predict_one_probas(tweet) for tweet in X])\n    \n### Haskell",
  temperature=0,
  max_tokens=54,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["###"]
)

"""
Advanced tweet classifier
Advanced sentiment detection for a piece of text.

@Tag:Classification
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Classify the sentiment in these tweets:\n\n1. \"I can't stand homework\"\n2. \"This sucks. I'm bored 😠\"\n3. \"I can't wait for Halloween!!!\"\n4. \"My cat is adorable ❤️❤️\"\n5. \"I hate chocolate\"\n\nTweet sentiment ratings:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Explain code
Explain a complicated piece of code.

@Tag:Code, @Tag:Translation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="class Log:\n    def __init__(self, path):\n        dirname = os.path.dirname(path)\n        os.makedirs(dirname, exist_ok=True)\n        f = open(path, \"a+\")\n\n        # Check that the file is newline-terminated\n        size = os.path.getsize(path)\n        if size > 0:\n            f.seek(size - 1)\n            end = f.read(1)\n            if end != \"\\n\":\n                f.write(\"\\n\")\n        self.f = f\n        self.path = path\n\n    def log(self, event):\n        event[\"_event_id\"] = str(uuid.uuid4())\n        json.dump(event, self.f)\n        self.f.write(\"\\n\")\n\n    def state(self):\n        state = {\"complete\": set(), \"last\": None}\n        for line in open(self.path):\n            event = json.loads(line)\n            if event[\"type\"] == \"submit\" and event[\"success\"]:\n                state[\"complete\"].add(event[\"id\"])\n                state[\"last\"] = event\n        return state\n\n\"\"\"\nHere's what the above class is doing:\n1.",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\"\"\""]
)

"""
Keywords
Extract keywords from a block of text.

@Tag:Classification, @Tag:Transformation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Extract keywords from this text:\n\nBlack-on-black ware is a 20th- and 21st-century pottery tradition developed by the Puebloan Native American ceramic artists in Northern New Mexico. Traditional reduction-fired blackware has been made for centuries by pueblo artists. Black-on-black ware of the past century is produced with a smooth surface, with the designs applied through selective burnishing or the application of refractory slip. Another style involves carving or incising designs and selectively polishing the raised areas. For generations several families from Kha'po Owingeh and P'ohwhóge Owingeh pueblos have been making black-on-black ware with the techniques passed down from matriarch potters. Artists from other pueblos have also produced black-on-black ware. Several contemporary artists have created works honoring the pottery of their ancestors.",
  temperature=0.3,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.8,
  presence_penalty=0.0
)

"""
Factual answering
Guide the model towards factual answering by showing it how to respond to questions that fall outside its knowledge base. Using a '?' to indicate a response to words and phrases that it doesn't know provides a natural response that seems to work better than more abstract replies.

@Tag:Answers, @Tag:Generation, @Tag:Conversation, @Tag:Classification
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Q: Who is Batman?\nA: Batman is a fictional comic book character.\n\nQ: What is torsalplexity?\nA: ?\n\nQ: What is Devz9?\nA: ?\n\nQ: Who is George Lucas?\nA: George Lucas is American film director and producer famous for creating Star Wars.\n\nQ: What is the capital of California?\nA: Sacramento.\n\nQ: What orbits the Earth?\nA: The Moon.\n\nQ: Who is Fred Rickerson?\nA: ?\n\nQ: What is an atom?\nA: An atom is a tiny particle that makes up everything.\n\nQ: Who is Alvan Muntz?\nA: ?\n\nQ: What is Kozar-09?\nA: ?\n\nQ: How many moons does Mars have?\nA: Two, Phobos and Deimos.\n\nQ: What's a language model?\nA:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Ad from product description
Turn a product description into ad copy.

@Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Write a creative ad for the following product to run on Facebook aimed at parents:\n\nProduct: Learning Room is a virtual environment to help students from kindergarten to high school excel in school.",
  temperature=0.5,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Product name generator
Create product names from examples words. Influenced by a community prompt.

@Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Product description: A home milkshake maker\nSeed words: fast, healthy, compact.\nProduct names: HomeShaker, Fit Shaker, QuickShake, Shake Maker\n\nProduct description: A pair of shoes that can fit any foot size.\nSeed words: adaptable, fit, omni-fit.",
  temperature=0.8,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
TL;DR summarization
Summarize text by adding a 'tl;dr:' to the end of a text passage. It shows that the API understands how to perform a number of tasks with no instructions.

@Tag:Transformation, @Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="A neutron star is the collapsed core of a massive supergiant star, which had a total mass of between 10 and 25 solar masses, possibly more if the star was especially metal-rich.[1] Neutron stars are the smallest and densest stellar objects, excluding black holes and hypothetical white holes, quark stars, and strange stars.[2] Neutron stars have a radius on the order of 10 kilometres (6.2 mi) and a mass of about 1.4 solar masses.[3] They result from the supernova explosion of a massive star, combined with gravitational collapse, that compresses the core past white dwarf star density to that of atomic nuclei.\n\nTl;dr",
  temperature=0.7,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Python bug fixer
Find and fix bugs in source code.

@Tag:Code, @Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="##### Fix bugs in the below function\n \n### Buggy Python\nimport Random\na = random.randint(1,12)\nb = random.randint(1,12)\nfor i in range(10):\n    question = \"What is \"+a+\" x \"+b+\"? \"\n    answer = input(question)\n    if answer = a*b\n        print (Well done!)\n    else:\n        print(\"No.\")\n    \n### Fixed Python",
  temperature=0,
  max_tokens=182,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["###"]
)

"""
Spreadsheet creator
Create spreadsheets of various kinds of data. It's a long prompt but very versatile. Output can be copy+pasted into a text file and saved as a .csv with pipe separators.

@Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="A two-column spreadsheet of top science fiction movies and the year of release:\n\nTitle|  Year of release",
  temperature=0.5,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)


"""
JavaScript helper chatbot
Message-style bot that answers JavaScript questions

@Tag:Code, @Tag:Answers, @Tag:Conversation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="You: How do I combine arrays?\nJavaScript chatbot: You can use the concat() method.\nYou: How do you make an alert appear after 10 seconds?\nJavaScript chatbot",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0,
  stop=["You:"]
)


"""
ML/AI language model tutor
Bot that answers questions about language models

@Tag:Answers, @Tag:Generation, @Tag:Conversation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="ML Tutor: I am a ML/AI language model tutor\nYou: What is a language model?\nML Tutor: A language model is a statistical model that describes the probability of a word given the previous words.\nYou: What is a statistical model?",
  temperature=0.3,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0,
  stop=["You:"]
)

"""
Science fiction book list maker
Create a list of items for a given topic.

@Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="List 10 science fiction books:",
  temperature=0.5,
  max_tokens=200,
  top_p=1.0,
  frequency_penalty=0.52,
  presence_penalty=0.5,
  stop=["11."]
)

"""
Tweet classifier
Basic sentiment detection for a piece of text.

@Tag:Classification
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Decide whether a Tweet's sentiment is positive, neutral, or negative.\n\nTweet: \"I loved the new Batman movie!\"\nSentiment:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0
)


"""
Airport code extractor
Extract airport codes from text.

@Tag:Transformation, @Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Extract the airport codes from this text:\n\nText: \"I want to fly from Los Angeles to Miami.\"\nAirport codes: LAX, MIA\n\nText: \"I want to fly from Orlando to Boston\"\nAirport codes:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)

"""
SQL request
Create simple SQL queries.

@Tag:Transformation, @Tag:Generation, @Tag:Translation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Create a SQL request to find all users who live in California and have over 1000 credits:",
  temperature=0.3,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Extract contact information
Extract contact information from a block of text.

@Tag:Transformation, @Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Extract the name and mailing address from this email:\n\nDear Kelly,\n\nIt was great to talk to you at the seminar. I thought Jane's talk was quite good.\n\nThank you for the book. Here's my address 2111 Ash Lane, Crestview CA 92002\n\nBest,\n\nMaya\n\nName:",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
JavaScript to Python
Convert simple JavaScript expressions into Python.

@Tag:Code, @Tag:Transformation, @Tag:Translation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="#JavaScript to Python:\nJavaScript: \ndogs = [\"bill\", \"joe\", \"carl\"]\ncar = []\ndogs.forEach((dog) {\n    car.push(dog);\n});\n\nPython:",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Friend chat
Emulate a text message conversation.

@Tag:Conversation, @Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="You: What have you been up to?\nFriend: Watching old movies.\nYou: Did you watch anything interesting?\nFriend:",
  temperature=0.5,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0,
  stop=["You:"]
)

"""
Mood to color
Turn a text description into a color.

@Tag:Translation, @Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="The CSS code for a color like a blue sky at dusk:\n\nbackground-color: #",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=[";"]
)


"""

Write a Python docstring
An example of how to create a docstring for a given Python function. We specify the Python version, paste in the code, and then ask within a comment for a docstring, and give a characteristic beginning of a docstring (triple-quotes).

@Tag:Code, @Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="# Python 3.7\n \ndef randomly_split_dataset(folder, filename, split_ratio=[0.8, 0.2]):\n    df = pd.read_json(folder + filename, lines=True)\n    train_name, test_name = \"train.jsonl\", \"test.jsonl\"\n    df_train, df_test = train_test_split(df, test_size=split_ratio[1], random_state=42)\n    df_train.to_json(folder + train_name, orient='records', lines=True)\n    df_test.to_json(folder + test_name, orient='records', lines=True)\nrandomly_split_dataset('finetune_data/', 'dataset.jsonl')\n    \n# An elaborate, high quality docstring for the above function:\n\"\"\"",
  temperature=0,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["#", "\"\"\""]
)


"""
Analogy maker
Create analogies. Modified from a community prompt to require fewer examples.

@Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Create an analogy for this phrase:\n\nQuestions are arrows in that:",
  temperature=0.5,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
JavaScript one line function
Turn a JavaScript function into a one liner.

@Tag:Code, @Tag:Transformation, @Tag:Translation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="code-davinci-001",
  prompt="Use list comprehension to convert this into one line of JavaScript:\n\ndogs.forEach((dog) => {\n    car.push(dog);\n});\n\nJavaScript one line version:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=[";"]
)

"""
Micro horror story creator
Creates two to three sentence short horror stories from a topic input.

@Tag:Transformation, @Tag:Generation, @Tag:Translation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Topic: Breakfast\nTwo-Sentence Horror Story: He always stops crying when I pour the milk on his cereal. I just have to remember not to let him see his face on the carton.\n    \nTopic: Wind\nTwo-Sentence Horror Story:",
  temperature=0.8,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0
)

"""
Third-person converter
Converts first-person POV to the third-person. This is modified from a community prompt to use fewer examples.

@Tag:Transformation, @Tag:Generation, @Tag:Translation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Convert this from first-person to third person (gender female):\n\nI decided to make a movie about Ada Lovelace.",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Notes to summary
Turn meeting notes into a summary.

@Tag: Transformation, @Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Convert my short hand into a first-hand account of the meeting:\n\nTom: Profits up 50%\nJane: New servers are online\nKjel: Need more time to fix software\nJane: Happy to help\nParkman: Beta testing almost done",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
VR fitness idea generator
Create ideas for fitness and virtual reality games.

@Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Brainstorm some ideas combining VR and fitness:",
  temperature=0.6,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=1,
  presence_penalty=1
)

"""
ESRB rating
Categorize text based upon ESRB ratings.

@Tag:Classification
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Provide an ESRB rating for the following text:\n\n\"i'm going to blow your brains out with my ray gun then stomp on your guts.\"\n\nESRB rating:",
  temperature=0.3,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)


"""
Essay outline
Generate an outline for a research topic.

@Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Create an outline for an essay about Walt Disney and his contributions to animation:\n\nI:",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Recipe creator (eat at your own risk)
Create a recipe from a list of ingredients.

@Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Write a recipe based on these ingredients and instructions:\n\nFrito Pie\n\nIngredients:\nFritos\nChili\nShredded cheddar cheese\nSweet white or red onions, diced small\nSour cream\n\nInstructions:",
  temperature=0.3,
  max_tokens=120,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Chat
Open ended conversation with an AI assistant.

@Tag:Conversation, @Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:",
  temperature=0.9,
  max_tokens=150,
  top_p=1,
  frequency_penalty=0.0,
  presence_penalty=0.6,
  stop=[" Human:", " AI:"]
)

"""
Marv the sarcastic chat bot
Marv is a factual chatbot that is also sarcastic.

@Tag:Conversation, @Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Marv is a chatbot that reluctantly answers questions with sarcastic responses:\n\nYou: How many pounds are in a kilogram?\nMarv: This again? There are 2.2 pounds in a kilogram. Please make a note of this.\nYou: What does HTML stand for?\nMarv: Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future.\nYou: When did the first airplane fly?\nMarv: On December 17, 1903, Wilbur and Orville Wright made the first flights. I wish they’d come and take me away.\nYou: What is the meaning of life?\nMarv: I’m not sure. I’ll ask my friend Google.\nYou: What time is it?\nMarv:",
  temperature=0.5,
  max_tokens=60,
  top_p=0.3,
  frequency_penalty=0.5,
  presence_penalty=0.0
)

"""
Turn by turn directions
Convert natural language to turn-by-turn directions.

@Tag:Transformation, @Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Create a numbered list of turn-by-turn directions from this text: \n\nGo south on 95 until you hit Sunrise boulevard then take it east to us 1 and head south. Tom Jenkins bbq will be on the left after several miles.",
  temperature=0.3,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Restaurant review creator
Turn a few words into a restaurant review.

@Tag:Generation
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Write a restaurant review based on these notes:\n\nName: The Blue Wharf\nLobster great, noisy, service polite, prices good.\n\nReview:",
  temperature=0.5,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)


"""
Create study notes
Provide a topic and get study notes.

@Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="What are 5 key points I should know when studying Ancient Rome?",
  temperature=0.3,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

"""
Interview questions
Create interview questions.

@Tag:Generation
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Create a list of 8 questions for my interview with a science fiction author:",
  temperature=0.5,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
