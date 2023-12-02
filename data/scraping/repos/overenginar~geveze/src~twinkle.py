import openai
import os
import click

# interview_questions
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_LOOKUP = {
    "product_description": {
        "group": "language",
        "description": "Ad from product description \n Turn a product description into ad copy.",
        "tags": ["Generation"],
        "prompt": "Write a creative ad for the following product to run on Facebook aimed at parents:\n\nProduct: Learning Room is a virtual environment to help students from kindergarten to high school excel in school.",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.5,
            "max_tokens": 100,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "product_name_generator": {
        "group": "language",
        "description": "Product name generator \n Create product names from examples words. Influenced by a community prompt.",
        "tags": ["Generation"],
        "prompt": "Product description: A home milkshake maker\nSeed words: fast, healthy, compact.\nProduct names: HomeShaker, Fit Shaker, QuickShake, Shake Maker\n\nProduct description: A pair of shoes that can fit any foot size.\nSeed words: adaptable, fit, omni-fit.",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.8,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "advanced_twitter_classifier": {
        "group": "language",
        "description": "Advanced tweet classifier \n This is an advanced prompt for detecting sentiment. It allows you to provide it with a list of status updates and then provide a sentiment for each one.",
        "tags": ["Classification"],
        "prompt": 'Classify the sentiment in these tweets:\n\n1. "I can\'t stand homework"\n2. "This sucks. I\'m bored 😠"\n3. "I can\'t wait for Halloween!!!"\n4. "My cat is adorable ❤️❤️"\n5. "I hate chocolate"\n\nTweet sentiment ratings:',
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "classification": {
        "group": "language",
        "description": "Classification \n Classify items into categories via example.",
        "tags": ["Classification"],
        "prompt": "The following is a list of companies and the categories they fall into:\n\nApple, Facebook, Fedex\n\nApple\nCategory:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "translate": {
        "group": "language",
        "description": "English to other languages \n Translates English text into French, Spanish and Japanese.",
        "tags": ["Transformation", "Generation"],
        "prompt": "Translate this into 1. French, 2. Spanish and 3. Japanese:\n\nWhat rooms do you have available?\n\n1.",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.3,
            "max_tokens": 100,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "factual_answering": {
        "group": "language",
        "description": "Factual answering \n Guide the model towards factual answering by showing it how to respond to questions that fall outside its knowledge base. Using a '?' to indicate a response to words and phrases that it doesn't know provides a natural response that seems to work better than more abstract replies.",
        "tags": ["Answers", "Generation", "Conversation", "Classification"],
        "prompt": "Q: Who is Batman?\nA: Batman is a fictional comic book character.\n\nQ: What is torsalplexity?\nA: ?\n\nQ: What is Devz9?\nA: ?\n\nQ: Who is George Lucas?\nA: George Lucas is American film director and producer famous for creating Star Wars.\n\nQ: What is the capital of California?\nA: Sacramento.\n\nQ: What orbits the Earth?\nA: The Moon.\n\nQ: Who is Fred Rickerson?\nA: ?\n\nQ: What is an atom?\nA: An atom is a tiny particle that makes up everything.\n\nQ: Who is Alvan Muntz?\nA: ?\n\nQ: What is Kozar-09?\nA: ?\n\nQ: How many moons does Mars have?\nA: Two, Phobos and Deimos.\n\nQ: What's a language model?\nA:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "grammar_correction": {
        "group": "language",
        "description": "Grammar correction \n Corrects sentences into standard English.",
        "tags": ["Transformation", "Generation"],
        "prompt": "Correct this to standard English:\n\nShe no went to the market.",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "keywords": {
        "group": "language",
        "description": "Keywords \n Extract keywords from a block of text. At a lower temperature it picks keywords from the text. At a higher temperature it will generate related keywords which can be helpful for creating search indexes.",
        "tags": ["Classification", "Transformation"],
        "prompt": "Extract keywords from this text:\n\nBlack-on-black ware is a 20th- and 21st-century pottery tradition developed by the Puebloan Native American ceramic artists in Northern New Mexico. Traditional reduction-fired blackware has been made for centuries by pueblo artists. Black-on-black ware of the past century is produced with a smooth surface, with the designs applied through selective burnishing or the application of refractory slip. Another style involves carving or incising designs and selectively polishing the raised areas. For generations several families from Kha'po Owingeh and P'ohwhóge Owingeh pueblos have been making black-on-black ware with the techniques passed down from matriarch potters. Artists from other pueblos have also produced black-on-black ware. Several contemporary artists have created works honoring the pottery of their ancestors.",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.5,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.8,
            "presence_penalty": 0.0,
        },
    },
    "lang_model_tutor": {
        "group": "language",
        "description": "ML/AI language model tutor. \n This is a QA-style chatbot that answers questions about language models.",
        "tags": ["Answers", "Generation", "Conversation"],
        "prompt": "ML Tutor: I am a ML/AI language model tutor\nYou: What is a language model?\nML Tutor: A language model is a statistical model that describes the probability of a word given the previous words.\nYou: What is a statistical model?",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.3,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.0,
            "stop": ["You:"],
        },
    },
    "movie_to_emoji": {
        "group": "language",
        "description": "Movie to Emoji \n Convert movie titles into emoji.",
        "tags": ["Transformation", "Generation"],
        "prompt": "Convert movie titles into emoji.\n\nBack to the Future: 👨👴🚗🕒 \nBatman: 🤵🦇 \nTransformers: 🚗🤖 \nStar Wars:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.8,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["\n"],
        },
    },
    "qa": {
        "group": "language",
        "description": "Q&A \n Answer questions based on existing knowledge.",
        "tags": ["Answers", "Generation", "Conversation"],
        "prompt": 'I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with "Unknown".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ: Where is the Valley of Kings?\nA:',
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 100,
            "top_p": 1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["\n"],
        },
    },
    "science_fiction_book_list_maker": {
        "group": "language",
        "description": "Science fiction book list maker \n This makes a list of science fiction books and stops when it reaches #10.",
        "tags": ["Generation"],
        "prompt": "List 10 science fiction books:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.5,
            "max_tokens": 200,
            "top_p": 1.0,
            "frequency_penalty": 0.52,
            "presence_penalty": 0.5,
            "stop": ["11."],
        },
    },
    "summarise_for_2nd_grader": {
        "group": "language",
        "description": "Summarize for a 2nd grader \n Translates difficult text into simpler concepts.",
        "tags": ["Transformation", "Generation"],
        "prompt": "Summarize this for a second-grade student:\n\nJupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.7,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "tldr_summarisation": {
        "group": "language",
        "description": "TL;DR summarization \n Summarize text by adding a 'tl;dr:' to the end of a text passage. It shows that the API understands how to perform a number of tasks with no instructions.",
        "tags": ["Transformation", "Generation"],
        "prompt": "A neutron star is the collapsed core of a massive supergiant star, which had a total mass of between 10 and 25 solar masses, possibly more if the star was especially metal-rich.[1] Neutron stars are the smallest and densest stellar objects, excluding black holes and hypothetical white holes, quark stars, and strange stars.[2] Neutron stars have a radius on the order of 10 kilometres (6.2 mi) and a mass of about 1.4 solar masses.[3] They result from the supernova explosion of a massive star, combined with gravitational collapse, that compresses the core past white dwarf star density to that of atomic nuclei.\n\nTl;dr",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.7,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 1,
        },
    },
    "tweet_classifier": {
        "group": "language",
        "description": "Tweet classifier \n This is a basic prompt for detecting sentiment.",
        "tags": ["Classification"],
        "prompt": 'Decide whether a Tweet\'s sentiment is positive, neutral, or negative.\n\nTweet: "I loved the new Batman movie!"\nSentiment:',
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.0,
        },
    },
    "airport_code_extractor": {
        "group": "language",
        "description": "Tweet classifier \n This is a basic prompt for detecting sentiment.",
        "tags": ["Classification"],
        "prompt": 'Extract the airport codes from this text:\n\nText: "I want to fly from Los Angeles to Miami."\nAirport codes: LAX, MIA\n\nText: "I want to fly from Orlando to Boston"\nAirport codes:',
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["\n"],
        },
    },
    "extract_contact_information": {
        "group": "language",
        "description": "Extract contact information \n Extract contact information from a block of text.",
        "tags": ["Transformation", "Generation"],
        "prompt": "Extract the name and mailing address from this email:\n\nDear Kelly,\n\nIt was great to talk to you at the seminar. I thought Jane's talk was quite good.\n\nThank you for the book. Here's my address 2111 Ash Lane, Crestview CA 92002\n\nBest,\n\nMaya\n\nName:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "friend_chat": {
        "group": "language",
        "description": "Friend chat \n Emulate a text message conversation.",
        "tags": ["Conversation", "Generation"],
        "prompt": "You: What have you been up to?\nFriend: Watching old movies.\nYou: Did you watch anything interesting?\nFriend:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.5,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.0,
            "stop": ["You:"],
        },
    },
    "analogy_maker": {
        "group": "language",
        "description": "Analogy maker \n Create analogies. Modified from a community prompt to require fewer examples.",
        "tags": ["Generation"],
        "prompt": "Create an analogy for this phrase:\n\nQuestions are arrows in that:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.5,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "micro_horror_story_creator": {
        "group": "language",
        "description": "Micro horror story creator \n Creates two to three sentence short horror stories from a topic input.",
        "tags": ["Transformation", "Generation", "Translation"],
        "prompt": "Topic: Breakfast\nTwo-Sentence Horror Story: He always stops crying when I pour the milk on his cereal. I just have to remember not to let him see his face on the carton.\n    \nTopic: Wind\nTwo-Sentence Horror Story:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.8,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.0,
        },
    },
    "third_person_converter": {
        "group": "language",
        "description": "Third-person converter \n Converts first-person POV to the third-person. This is modified from a community prompt to use fewer examples.",
        "tags": ["Transformation", "Generation", "Translation"],
        "prompt": "Convert this from first-person to third person (gender female):\n\nI decided to make a movie about Ada Lovelace.",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "notes_to_summary": {
        "group": "language",
        "description": "Notes to summary \n Turn meeting notes into a summary.",
        "tags": ["Transformation", "Generation"],
        "prompt": "Convert my short hand into a first-hand account of the meeting:\n\nTom: Profits up 50%\nJane: New servers are online\nKjel: Need more time to fix software\nJane: Happy to help\nParkman: Beta testing almost done",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "vr_fitness_idea_generator": {
        "group": "language",
        "description": "VR fitness idea generator \n Create ideas for fitness and virtual reality games.",
        "tags": ["Generation"],
        "prompt": "Brainstorm some ideas combining VR and fitness:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.6,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 1,
            "presence_penalty": 1,
        },
    },
    "essay_outline": {
        "group": "language",
        "description": "Essay outline \n Generate an outline for a research topic.",
        "tags": ["Generation"],
        "prompt": "Create an outline for an essay about Nikola Tesla and his contributions to technology:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.3,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "recipe_creator": {
        "group": "language",
        "description": "Recipe creator (eat at your own risk) \n Create a recipe from a list of ingredients.",
        "tags": ["Generation"],
        "prompt": "Write a recipe based on these ingredients and instructions:\n\nFrito Pie\n\nIngredients:\nFritos\nChili\nShredded cheddar cheese\nSweet white or red onions, diced small\nSour cream\n\nInstructions:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.3,
            "max_tokens": 120,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "chat": {
        "group": "language",
        "description": "Chat \n Open ended conversation with an AI assistant.",
        "tags": ["Conversation", "Generation"],
        "prompt": "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.9,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.6,
            "stop": [" Human:", " AI:"],
        },
    },
    "sarcastic_chatbot": {
        "group": "language",
        "description": "Marv the sarcastic chat bot \n Marv is a factual chatbot that is also sarcastic.",
        "tags": ["Conversation", "Generation"],
        "prompt": "Marv is a chatbot that reluctantly answers questions with sarcastic responses:\n\nYou: How many pounds are in a kilogram?\nMarv: This again? There are 2.2 pounds in a kilogram. Please make a note of this.\nYou: What does HTML stand for?\nMarv: Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future.\nYou: When did the first airplane fly?\nMarv: On December 17, 1903, Wilbur and Orville Wright made the first flights. I wish they’d come and take me away.\nYou: What is the meaning of life?\nMarv: I’m not sure. I’ll ask my friend Google.\nYou: What time is it?\nMarv:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.5,
            "max_tokens": 60,
            "top_p": 0.3,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.0,
        },
    },
    "turn_by_turn_directions": {
        "group": "language",
        "description": "Turn by turn directions \n Convert natural language to turn-by-turn directions.",
        "tags": ["Transformation", "Generation"],
        "prompt": "Create a numbered list of turn-by-turn directions from this text: \n\nGo south on 95 until you hit Sunrise boulevard then take it east to us 1 and head south. Tom Jenkins bbq will be on the left after several miles.",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.3,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "restaurant_review_creator": {
        "group": "language",
        "description": "Restaurant review creator \n Turn a few words into a restaurant review.",
        "tags": ["Generation"],
        "prompt": "Write a restaurant review based on these notes:\n\nName: The Blue Wharf\nLobster great, noisy, service polite, prices good.\n\nReview:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.5,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "create_study_notes": {
        "group": "language",
        "description": "Create study notes \n Provide a topic and get study notes.",
        "tags": ["Generation"],
        "prompt": "What are 5 key points I should know when studying Ancient Rome?",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.3,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "interview_questions": {
        "group": "language",
        "description": "Interview questions \n Create interview questions.",
        "tags": ["Generation"],
        "prompt": "Create a list of 8 questions for my interview with a science fiction author:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.5,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "time_complexity": {
        "group": "programming",
        "description": "Calculate Time Complexity \n Find the time complexity of a function.",
        "tags": ["Code", "Transformation"],
        "prompt": 'def foo(n, k):\naccum = 0\nfor i in range(n):\n    for l in range(k):\n        accum += i\nreturn accum\n"""\nThe time complexity of this function is',
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["\n"],
        },
    },
    "explain_code": {
        "group": "programming",
        "description": "Explain code \n Explain a complicated piece of code.",
        "tags": ["Code", "Translation"],
        "prompt": 'class Log:\n    def __init__(self, path):\n        dirname = os.path.dirname(path)\n        os.makedirs(dirname, exist_ok=True)\n        f = open(path, "a+")\n\n        # Check that the file is newline-terminated\n        size = os.path.getsize(path)\n        if size > 0:\n            f.seek(size - 1)\n            end = f.read(1)\n            if end != "\\n":\n                f.write("\\n")\n        self.f = f\n        self.path = path\n\n    def log(self, event):\n        event["_event_id"] = str(uuid.uuid4())\n        json.dump(event, self.f)\n        self.f.write("\\n")\n\n    def state(self):\n        state = {"complete": set(), "last": None}\n        for line in open(self.path):\n            event = json.loads(line)\n            if event["type"] == "submit" and event["success"]:\n                state["complete"].add(event["id"])\n                state["last"] = event\n        return state\n\n"""\nHere\'s what the above class is doing:\n1.',
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ['"""'],
        },
    },
    "javascript_helper": {
        "group": "programming",
        "description": "JavaScript helper chatbot \n This is a message-style chatbot that can answer questions about using JavaScript. It uses a few examples to get the conversation started.",
        "tags": ["Code", "Answers", "Conversation"],
        "prompt": "You: How do I combine arrays?\nJavaScript chatbot: You can use the concat() method.\nYou: How do you make an alert appear after 10 seconds?\nJavaScript chatbot",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.0,
            "stop": ["You:"],
        },
    },
    "nl_to_openai_api": {
        "group": "programming",
        "description": "Natural language to OpenAI API \n Create code to call to the OpenAI API using a natural language instruction.",
        "tags": ["Code", "Transformation"],
        "prompt": '"""\nUtil exposes the following:\nutil.openai() -> authenticates & returns the openai module, which has the following functions:\nopenai.Completion.create(\n    prompt="<my prompt>", # The prompt to start completing from\n    max_tokens=123, # The max number of tokens to generate\n    temperature=1.0 # A measure of randomness\n    echo=True, # Whether to return the prompt in addition to the generated completion\n)\n"""\nimport util\n"""\nCreate an OpenAI completion starting from the prompt "Once upon an AI", no more than 5 tokens. Does not include the prompt.\n"""\n',
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ['"""'],
        },
    },
    "nl_to_stripe_api": {
        "group": "programming",
        "description": "Natural language to Stripe API \n Create code to call the Stripe API using natural language.",
        "tags": ["Code", "Transformation"],
        "prompt": '"""\nUtil exposes the following:\n\nutil.stripe() -> authenticates & returns the stripe module; usable as stripe.Charge.create etc\n"""\nimport util\n"""\nCreate a Stripe token using the users credit card: 5555-4444-3333-2222, expiration date 12 / 28, cvc 521\n"""',
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 100,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ['"""'],
        },
    },
    "parse_unstructured_data": {
        "group": "programming",
        "description": "Parse unstructured data \n Create tables from long form text by specifying a structure and supplying some examples.",
        "tags": ["Transformation", "Generation"],
        "prompt": "A table summarizing the fruits from Goocrux:\n\nThere are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy. There are also loheckles, which are a grayish blue fruit and are very tart, a little bit like a lemon. Pounits are a bright green color and are more savory than sweet. There are also plenty of loopnovas which are a neon pink flavor and taste like cotton candy. Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them.\n\n| Fruit | Color | Flavor |",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 100,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "python_bug_fixer": {
        "group": "programming",
        "description": "Python bug fixer \n There's a number of ways of structuring the prompt for checking for bugs. Here we add a comment suggesting that source code is buggy, and then ask codex to generate a fixed code.",
        "tags": ["Code", "Generation"],
        "prompt": '##### Fix bugs in the below function\n \n### Buggy Python\nimport Random\na = random.randint(1,12)\nb = random.randint(1,12)\nfor i in range(10):\n    question = "What is "+a+" x "+b+"? "\n    answer = input(question)\n    if answer = a*b\n        print (Well done!)\n    else:\n        print("No.")\n    \n### Fixed Python',
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 182,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["###"],
        },
    },
    "python_to_nl": {
        "group": "programming",
        "description": "Python to natural language \n Explain a piece of Python code in human understandable language.",
        "tags": ["Code", "Translation"],
        "prompt": '# Python 3 \ndef remove_common_prefix(x, prefix, ws_prefix): \n    x["completion"] = x["completion"].str[len(prefix) :] \n    if ws_prefix: \n        # keep the single whitespace as prefix \n        x["completion"] = " " + x["completion"] \nreturn x \n\n# Explanation of what the code does\n\n#',
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "spreadsheet_creator": {
        "group": "programming",
        "description": "Spreadsheet creator \n Create spreadsheets of various kinds of data. It's a long prompt but very versatile. Output can be copy+pasted into a text file and saved as a .csv with pipe separators.",
        "tags": ["Generation"],
        "prompt": "A two-column spreadsheet of top science fiction movies and the year of release:\n\nTitle |  Year of release",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.5,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "sql_translate": {
        "group": "programming",
        "description": "SQL translate \n Translate natural language to SQL queries.",
        "tags": ["Code", "Transformation"],
        "prompt": "### Postgres SQL tables, with their properties:\n#\n# Employee(id, name, department_id)\n# Department(id, name, address)\n# Salary_Payments(id, employee_id, amount, date)\n#\n### A query to list the names of the departments which employed more than 10 employees in the last 3 months\nSELECT",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["#", ";"],
        },
    },
    "text_to_command": {
        "group": "programming",
        "description": "Text to command \n Translate text into programmatic commands.",
        "tags": ["Transformation", "Generation"],
        "prompt": "Convert this text to a programmatic command:\n\nExample: Ask Constance if we need some bread\nOutput: send-msg `find constance` Do we need some bread?\n\nReach out to the ski store and figure out if I can get my skis fixed before I leave on Thursday",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 100,
            "top_p": 1.0,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.0,
            "stop": ["\n"],
        },
    },
    "translate_programming_language": {
        "group": "programming",
        "description": "Translate programming languages \n To translate from one programming language to another we can use the comments to specify the source and target languages.",
        "tags": ["Code", "Translation"],
        "prompt": "##### Translate this function  from Python into Haskell\n### Python\n    \n    def predict_proba(X: Iterable[str]):\n        return np.array([predict_one_probas(tweet) for tweet in X])\n    \n### Haskell",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 54,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["###"],
        },
    },
    "sql_request": {
        "group": "programming",
        "description": "SQL request \n Create simple SQL queries.",
        "tags": ["Transformation", "Generation", "Translation"],
        "prompt": "Create a SQL request to find all users who live in California and have over 1000 credits:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0.3,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "js_to_python": {
        "group": "programming",
        "description": "JavaScript to Python \n Convert simple JavaScript expressions into Python.",
        "tags": ["Code", "Transformation", "Translation"],
        "prompt": '#JavaScript to Python:\nJavaScript: \ndogs = ["bill", "joe", "carl"]\ncar = []\ndogs.forEach((dog) {\n    car.push(dog);\n});\n\nPython:',
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "mood_to_color": {
        "group": "programming",
        "description": "Mood to color \n Turn a text description into a color.",
        "tags": ["Transformation", "Generation"],
        "prompt": "The CSS code for a color like a blue sky at dusk:\n\nbackground-color: #",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 64,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": [";"],
        },
    },
    "write_python_docstring": {
        "group": "programming",
        "description": 'Write a Python docstring \n An example of how to create a docstring for a given Python function. We specify the Python version, paste in the code, and then ask within a comment for a docstring, and give a characteristic beginning of a docstring (""").',
        "tags": ["Code", "Generation"],
        "prompt": "# Python 3.7\n \ndef randomly_split_dataset(folder, filename, split_ratio=[0.8, 0.2]):\n    df = pd.read_json(folder + filename, lines=True)\n    train_name, test_name = \"train.jsonl\", \"test.jsonl\"\n    df_train, df_test = train_test_split(df, test_size=split_ratio[1], random_state=42)\n    df_train.to_json(folder + train_name, orient='records', lines=True)\n    df_test.to_json(folder + test_name, orient='records', lines=True)\nrandomly_split_dataset('finetune_data/', 'dataset.jsonl')\n    \n# An elaborate, high quality docstring for the above function:\n\"\"\"",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["#", '"""'],
        },
    },
    "js_one_line_function": {
        "group": "programming",
        "description": "JavaScript one line function \n Turn a JavaScript function into a one liner.",
        "tags": ["Code", "Transformation", "Translation"],
        "prompt": "Use list comprehension to convert this into one line of JavaScript:\n\ndogs.forEach((dog) => {\n    car.push(dog);\n});\n\nJavaScript one line version:",
        "model": "text-davinci-003",
        "model_params": {
            "temperature": 0,
            "max_tokens": 60,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": [";"],
        },
    },
}


@click.group()
def cli():
    pass


@click.command()
def groups():
    """List of groups"""
    groups = set([value['group'] for category, value in PROMPT_LOOKUP.items()])
    for g in groups:
        click.echo(f"{g}")


@click.command()
@click.option("-g", "--group", prompt="Group", help="Find out categories: 'python -m src.assistant groups'", prompt_required=True)
def categories(group):
    """List of categories by groups"""
    for category, value in PROMPT_LOOKUP.items():
        if value['group'] == group.lower():
            click.echo(f"{category}:\t{value['description']}\n")
    else:
        click.echo(f"Key not found: {group}, please find a group using 'python -m src.assistant groups'")


@click.command()
@click.option("--category", prompt="Category", help="Find out categories: 'python -m src.assistant categories --group <Group Code>'", prompt_required=True)
def examples(category):
    """Simple program that makes requests for language API methods"""
    category_obj = PROMPT_LOOKUP.get(category.lower())
    if category_obj is None:
        click.echo(f"Key not found: {category}, please find a category using 'python -m src.assistant categories --group <Group Code>'")
        return
    response = openai.Completion.create(
        model=category_obj.get("model"),
        prompt=category_obj.get("prompt"),
        **category_obj.get("model_params"),
    )
    click.echo(f"\nDescription: {category_obj['description']}\n")
    click.echo(f"Tags: {', '.join(category_obj['tags'])}\n")
    click.echo(f"Prompt: {category_obj['prompt']}\n")
    click.echo(f"Model: {category_obj['model']}\n")
    click.echo(f"Model params: {category_obj['model_params']}\n")
    click.echo("Responses: ")
    for choice in response["choices"]:
        click.echo(f"{choice['text']}\n")


@click.command()
def surpriseme():
    """Simple program that makes requests for language API methods"""
    import random
    random_idx = random.randint(0, len(PROMPT_LOOKUP))
    random_category = list(PROMPT_LOOKUP.keys())[random_idx]
    category_obj = PROMPT_LOOKUP.get(random_category.lower())
    if category_obj is None:
        click.echo(f"Key not found: {random_category}, please find a category using 'python -m src.assistant categories --group <Group Code>'")
        return
    response = openai.Completion.create(
        model=category_obj.get("model"),
        prompt=category_obj.get("prompt"),
        **category_obj.get("model_params"),
    )
    click.echo(f"\nDescription: {category_obj['description']}\n")
    click.echo(f"Tags: {', '.join(category_obj['tags'])}\n")
    click.echo(f"Prompt: {category_obj['prompt']}\n")
    click.echo(f"Model: {category_obj['model']}\n")
    click.echo(f"Model params: {category_obj['model_params']}\n")
    click.echo("Responses: ")
    for choice in response["choices"]:
        click.echo(f"{choice['text']}\n")


@click.command()
@click.option("--prompt", prompt="Prompt", help="Prompt to request", prompt_required=True)
@click.option("--model", prompt="Model", default="text-davinci-003", help="Model: openai api models.list")
@click.option("--temperature", prompt="Temperature", default=0, help="Randomness: 0 < temperature <= 1, 0: deterministic, 1: random", type=float)
@click.option("--max_tokens", prompt="Max Tokens", default=50, help="Max Tokens", type=int)
@click.option("--top_p", prompt="Top P", default=1.0, help="Top P", type=float)
@click.option("--frequency_penalty", prompt="Frequency Penalty", default=0.0, help="Frequency Penalty", type=float)
@click.option("--presence_penalty", prompt="Presence Penalty", default=0.0, help="Presence Penalty", type=float)
def request(prompt, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    click.echo("Responses: ")
    for choice in response["choices"]:
        click.echo(f"{choice['text']}\n")


cli.add_command(groups)
cli.add_command(categories)
cli.add_command(examples)
cli.add_command(surpriseme)
cli.add_command(request)


if __name__ == "__main__":
    cli()
