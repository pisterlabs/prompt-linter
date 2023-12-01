import os
import openai
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# ------------------------------------------------------------
# Learning the completions endpoint
# ------------------------------------------------------------
# Testing it out
def ask_about_historical_figure(figure):
  prompt = f'''Tell me an important fact about this historical figure.

    Figure: George Washington
    Fact: First president of the United States.
    Figure:{figure}
    Fact: 
  '''

def return_prompt(prompt):
  return prompt

def do_completion_request(inp, make_prompt, max_tokens=128):
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=make_prompt(inp),
        temperature=.8,
        max_tokens=max_tokens,
    )
    return response['choices'][0]['text']

# test1 = do_completion_request("Suggest one name for a black horse.", return_prompt)
# print(test1)

'''
Notes:

Open AI has different models with differnt levels of quality and price
- Ada fucking sucks but it is cheap (use this for development!)
- Davinci is the best but expensive

Requests you make aren't free. Be careful!

Writing prompts is extremely important, for example:

  You can write this:
    "Suggest three names for a horse that is a superhero."

  Or you can write this:
    "Suggest three names for an animal that is a superhero.

    Animal: Cat
    Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
    Animal: Dog
    Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
    Animal: Horse
    Names:"

Stop treating like it's not an NLP tool. If you were asking a human, what would you ask?

'''

# ------------------------------------------------------------
# Playing with Temperature

'''
Notes:

If "temperature" is set to 0, which it is by default, the model will return the same thing each time.
The higher the temperature, the more 'risks' the model will take in its response, resulting in more diverse completions

'''

# ------------------------------------------------------------
# Building an application

'''
Notes:

They provide a quickstart guide... I downloaded the one for Flask. Fuck it. Time to learn.
'''

# ------------------------------------------------------------
# Reading The Other Docs

'''
Notes on the completion endpoint

ON COMPLETION
Prompt Design:
Our models can do everything from generating original stories to performing complex text analysis. 
Because they can do so many things, you have to be explicit in describing what you want. 
Showing, not just telling, is often the secret to a good prompt.

3 Basic Prompt Creating Guidelines
  1. Show and Tell, give instructions and examples
  2. Provide quality data
  3. Check your settings (Temperature & top_p). Is there only one right answer?

###############

ON CLASSIFICATION (still uses Completion API)
It still relies on prompts! For example:

  "Decide whether a Tweet's sentiment is positive, neutral, or negative.

  Tweet: I loved the new Batman movie!
  Sentiment:"

- Use plain language
- Give it options. Having a neutral option is important for Sentiment as well.
- You need fewer examples for familiar tasks. For this classifier, we don't provide any examples. 
  - This is because the API already has an understanding of sentiment and the concept of a Tweet. 
  - If you're building a classifier for something the API might not be familiar with, it might be necessary to provide more examples.

You can also do multiple requests at once, for example:

  "Classify the sentiment in these tweets:

  1. "I can't stand homework"
  2. "This sucks. I'm bored 😠"
  3. "I can't wait for Halloween!!!"
  4. "My cat is adorable ❤️❤️"
  5. "I hate chocolate"

  Tweet sentiment ratings:"

###############

ON GENERATION
You can generate new ideas with the API! This could be great for giving students smart things to say.
For example:

  "Brainstorm some ideas combining VR and fitness:"

###############

ON CONVERSATION
Here is an example:

  "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.

  Human: Hello, who are you?
  AI: I am an AI created by OpenAI. How can I help you today?
  Human:"

The key is that we also tell it HOW to behave, not just what to do!
  - We tell the API the intent but we also tell it how to behave.
  - We give the API an identity.

###############

ON TRANSLATION
You can ask it to translate things.

###############

ON SUMMARIZATION

For example:
  "Summarize this for a second-grade student:

  Jupiter is the fifth planet from the Sun and the largest in the Solar System.
  It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half 
  times that of all the other planets in the Solar System combined. Jupiter is one 
  of the brightest objects visible to the naked eye in the night sky, and has been 
  known to ancient civilizations since before recorded history. It is named after 
  the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough 
  for its reflected light to cast visible shadows,[20] and is on average the third-brightest 
  natural object in the night sky after the Moon and Venus."

This will be one of the most important things... but how much can it handle? What if you have a 10 page report?

I was able to ask it for an "Important quote from the passage." This was succesful, but I need to be more clear 
than just using the word "important."

ON PROVIDING FACTUAL RESPONSES

The API has a lot of knowlege, but it can make shit up too.
Supposedly there is a way to provide "ground truth" for the API.
  - Provide it with a body of test to answeer questions about
  - You can also tell it to say "I Don't Know" instead

'''

# ------------------------------------------------------------

'''
Notes on the Search Endpoint!

The search endpoint allows you do do a semantic search over a set of documents.
  - I.E. You can provide a query, such as a natural language question or a statement, and 
    the provided documents will be scored and ranked based on how semantically related they 
    are to the input query

!!!
Up to 200 documents can be passed as part of the request using the documents parameter.

One limitation to keep in mind is that the query and longest document must be below 2000 tokens together.
!!!

Imagine the possibilities. If you could split a document into paragraphs and rank them based on an
essay prompt, or a study question. This could be huge.
  - Student inputs a study question and their reading material
  - Tool returns the paragraphs most relevant to their question


'''


# ------------------------------------------------------------

'''
Notes on the Question Answering endpoint
https://beta.openai.com/docs/guides/answers

Here is the reference for the actual API:
https://beta.openai.com/docs/api-reference/answers

Text generation based on a list of 200 documents!
This is the one!

From the docs:
  "The endpoint first searches over provided documents or file to find relevant context for the input question. 
  Semantic search is used to rank documents by relevance to the question. The relevant context is combined with 
  the provided examples and question to create the prompt for completion."

'''

"ABOVE ARE ALL OF THE PREBUILT OPTIONS. IT IS INSANELY POWERFUL AND EASY TO USE!!! START DRAFTING PRODUCTS"
"A LITTLE NLP COMBINED WITH GOOD UX WILL GO A LONG WAY... IT'S ALL ABOUT THE DESIGN, THE MODEL IS STRONG!"
"It is like an employee, you still have to tell it how to help you. Sometimes that's the hardest part..."
"So now, you must think of how it can help you!"

# ------------------------------------------------------------

'''
Notes on fine tuning:

You can create a fine tuned model so you no longer need to provide context.
  1. Prepare and upload training data
  2. Train a new fine-tuned model
  3. Use your fine tuned model



TRAINING DATA: (https://beta.openai.com/docs/guides/fine-tuning/preparing-your-dataset)
- It must be in JSONL document
  - Each line is a prompt/completion pair:
      {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
      {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
      {"prompt": "<prompt text>", "completion": "<ideal generated text>"}

- In general, each doubling of the dataset size leads to a linear increase in model quality.

CREATING THE FINE-TUNED MODEL:

  `openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m <BASE_MODEL>`

  This command will:
    - Upload files to the Files API (https://beta.openai.com/docs/api-reference/files)
    - Create a fine-tune job
    - Streams events until the job is done (minutes to hours depending on dataset)
    - Build from the BASE_MODEL: ada, babbage, curie, or davinci

WHEN IT'S COMPLETE:
  - You may now specify this model as a parameter to the Completions API
  - You can start making requests by passing the model name as the `model` parameter of a completion request:

      import openai
      openai.Completion.create(
          model=FINE_TUNED_MODEL,
          prompt=YOUR_PROMPT)

  
####################################

More Specific & Tactical Notes:

PREPARING DATASET:


'''


# ------------------------------------------------------------



# ------------------------------------------------------------