"""
# values_conf.py.py should contain the following variables

# CAUTION
# do not forget to change it with your own values

# call in python file using VALUE = funk.function(a, b)


"""

TEXT_TITLE_APP = "Using OpenAI platform"
TEXT_SUBHEADER_APP = "Explore what's possible with some example applications with **OpenAI platform**"
TEXT_WARNING = 'It requires a license from OpenAI platform (paid) and Streamlit (free)'

TEXT_OUTPUT = '**OUTPUT**'

# HELP TEXTS
# help="help text"
TEXT_HELP_1 = 'The main navigation for the application'

# MAIN APP VALUES
LABEL_EXPANDER = "See explanation"

# 1. FOR DATA SCIENCE PURPOSES

# CASE_9
# 9. Parse unstructured data

# CASE_10
# 10. Classification

# CASE_3
# 3. Summarize for a 2nd grader

# INFOS
# **Summarize so it translates difficult text into simpler concepts.**
# **Example:**
# Canzone napoletana (pronounced[kanˈtsoːne napoleˈtaːna.Neapolitan: canzona napulitana[kanˈdzoːnə napuliˈtɑːnə]), sometimes referred to as Neapolitan song, is a generic term for a traditional form of music sung in the Neapolitan language, ordinarily for the male voice singing solo, although well represented by female soloists as well, and expressed in familiar genres such as the love song and serenade. Many of the songs are about the nostalgic longing for Naples as it once was. The genre consists of a large body of composed popular music—such songs as "’O sole mio" and others. The Neapolitan song became a formal institution in the 1830s due to an annual song-writing competition for the Festival of Piedigrotta, dedicated to the Madonna of Piedigrotta, a well-known church in the Mergellina area of Naples.

# INPUT
# Enter the text to summarize
# Summarize this for a second-grade student:
# **Set parameters:**
# Temperature
# Tokens

# OUTPUT
# Summarize
# Text summarized

# CASE_21
# 21. TL DR summarization


# 2. FOR PO PURPOSES

# 2.1 CODE MANAGEMENT
# CASE_11
# 11. Python to natural language

# CASE_22
# 22. Python bug fixer

# 2.2 PROJECT MANAGEMENT

# CASE_23
# 23. Spreadsheet creator

# CASE_6
# 6. English to other languages


# CASE_39
# 39. Notes to summary

# 3. FUN & PSEUDO ENTERTAINMENT
# CASE_12
# 12. Movie to Emoji(movie_to_emoji)



MENU_SIDEBAR_USECASE_TITLE_OPTIONS = ['INTRODUCTION', 'Q&A', 'Grammar correction', '3 :: Summarize for a 2nd grader', 'Natural language to OpenAI API', 'Text to command', '6 :: English to other languages', 'Natural language to Stripe API', 'SQL translate', '9 :: Parse unstructured data', '10 :: Classification', '11 :: Python to natural language', '12 :: Movie to Emoji', 'Calculate Time Complexity', 'Translate programming languages', 'Advanced tweet classifier', 'Explain code', '17 :: Keywords', 'Factual answering', 'Ad from product description', 'Product name generator', '21 :: TL;DR summarization', '22 :: Python bug fixer', '23 :: Spreadsheet creator',
                                      'JavaScript helper chatbot', 'ML/AI language model tutor.', 'Science fiction book list maker', 'Tweet classifier', 'Airport code extractor', 'SQL request', 'Extract contact information', 'JavaScript to Python', 'Friend chat', 'Mood to color', 'Write a Python docstring', 'Analogy maker', 'JavaScript one line function', 'Micro horror story creator', 'Third-person converter', '39 :: Notes to summary', 'VR fitness idea generator', 'Essay outline', 'Recipe creator (eat at your own risk)', 'Chat', 'Marv the sarcastic chat bot', 'Turn by turn directions', 'Restaurant review creator', 'Create study notes', 'Interview questions']

MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS = ['General description for OpenAI platform examples.', 'Answer questions based on existing knowledge.', 'Corrects sentences into standard English.', 'Translates difficult text into simpler concepts.', 'Create code to call to the OpenAI API using a natural language instruction.', 'Translate text into programmatic commands.', 'Translates English text into Spanish, Portuguese, Italian, French and Japanese.', 'Create code to call the Stripe API using natural language.', 'Translate natural language to SQL queries.', 'Create tables from long form text.', 'Classify items into categories via example.', 'Explain a piece of Python code in human understandable language.', 'Convert movie titles into emoji.', 'Find the time complexity of a function.', 'Translate from one programming language to another.', 'Advanced sentiment detection for a piece of text.', 'Explain a complicated piece of code.', 'Extract keywords from a block of text.', 'Direct the model to provide factual answers and address knowledge gaps.', 'Turn a product description into ad copy.', 'Create product names from examples words.', "Summarize text by adding a 'tl;dr:' to the end of a text passage. It shows that the API understands how to perform a number of tasks with no instructions.",
                                            'Find and fix bugs in source code.', 'Create spreadsheets of various kinds of data.', 'Message-style bot that answers JavaScript questions.', 'Bot that answers questions about language models', 'Create a list of items for a given topic.', 'Basic sentiment detection for a piece of text.', 'Extract airport codes from text.', 'Create simple SQL queries.', 'Extract contact information from a block of text.', 'Convert simple JavaScript expressions into Python.', 'Emulate a text message conversation.', 'Turn a text description into a color.', 'Write a docstring for a Python function.', 'Create analogies.', 'Turn a JavaScript function into a one liner.', 'Creates two to three sentence short horror stories from a topic input.', 'Converts first-person POV to the third-person.', 'Turn meeting notes into a summary.', 'Create ideas for fitness and virtual reality games.', 'Generate an outline for a research topic.', 'Create a recipe from a list of ingredients.', 'Open ended conversation with an AI assistant.', 'Marv is a factual chatbot that is also sarcastic.', 'Convert natural language to turn-by-turn directions.', 'Turn a few words into a restaurant review.', 'Provide a topic and get study notes.', 'Create interview questions.']




# MENU_SIDEBAR_OPTIONS[0]
# MENU_SIDEBAR_OPTIONS[1]
# MENU_SIDEBAR_OPTIONS[2]

TEXT_HELP_2 = 'The main navigation for the application'

GREAT_EXPENDER_TEXT_1 = 'GREAT_EXPENDER_TEXT_1'
GREAT_EXPENDER_TEXT_2 = 'GREAT_EXPENDER_TEXT_2'
GREAT_EXPENDER_TEXT_3 = 'GREAT_EXPENDER_TEXT_3'

TEXT_WARNING_REPORT = 'TEXT_WARNING_REPORT'
TEXT_WARNING_REPORT_HELP = "TEXT_WARNING_REPORT_HELP"


# USECASES

# CASE_6 :: 6. English to other languages
CASE_6_LANGUAGES_SELECTION = ['Spanish',
                                'Portuguese',
                                'Italian',
                                'French',
                                'Japanese']
TEXT_HELP_CASE_6_1 = 'Select one or more languages among the dropdown menu'
TEXT_HELP_CASE_6_2 = 'Enter just the text to translate'


TEXT_HELP_TEMPERATURE_PARAMETER = 'Temperature: number; Optional;Defaults to 1;What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both. More on https://platform.openai.com/docs/api-reference/completions'
TEXT_HELP_TOKENS_PARAMETER = 'Tokens: They are the fundamental units of text used by language models like ChatGPT to process and analyze text. Tokens can be single characters, words, or parts of words, depending on the tokenizer used and the language.. More on https://azurebrainwave.com/posts/tokens-in-chatgpt/'

# CASE_3 :: 3. Summarize for a 2nd grader
TEXT_HELP_CASE_3_1 = 'Enter just the text to summarize. It will go from difficult to simplest!'

# CASE_21 :: 21. TL DR summarization
TEXT_HELP_CASE_21_1 = 'Enter just the text to summarize. TL;DR or TLDR stands for "Too Long; Didn\'t Read." TLDR can be used to express that a text is too long, identify a short summary of a long text, or ask for a summary of a long text.'

# CASE_39 :: 39. Notes to summary
TEXT_HELP_CASE_39_1 = 'Enter just the notes to summarize. Ideal to write meeting minutes for a PO or a Project Manager.'

# CASE_23 :: 23. Spreadsheet creator
TEXT_HELP_CASE_23_1 = 'Create spreadsheets of various kinds of data. It\'s a long prompt but very versatile. Output can be copy+pasted into a text file and saved as a .csv with pipe separators.'

# CASE_22 :: 22. Python bug fixer
TEXT_HELP_CASE_22_1 = 'There\'s a number of ways of structuring the prompt for checking for bugs. Here we add a comment suggesting that source code is buggy, and then ask codex to generate a fixed code.'

# CASE_12 :: + 12. Movie to Emoji (movie_to_emoji)
TEXT_HELP_CASE_12_1 = 'Convert movie titles into emoji.'


