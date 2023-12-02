# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# # Using ChatGPT to label data
#
# - Fetch list of companies
# - Prepare prompt
# - Generate reponses and save to disk

# +
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

import innovation_sweet_spots.getters.google_sheets as gs
from innovation_sweet_spots.getters.preprocessed import (
    get_preprocessed_crunchbase_descriptions,
)
import utils

# +
# Get longlist of companies
longlist_df = gs.download_google_sheet(
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
    wks_name="list_v2",
)

# Load a table with processed company descriptions
processed_texts = get_preprocessed_crunchbase_descriptions()

# Get our taxonomy
taxonomy_df = gs.download_google_sheet(
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
    wks_name="taxonomy",
)
# -

subtheme_to_keywords_dict = utils.get_taxonomy_dict(taxonomy_df, "subtheme", "keywords")

subtheme_keywords_dict = utils.get_taxonomy_dict(taxonomy_df, "subtheme", "keywords")

chatgpt_theme_to_subthemes = {
    "Content: General": "General content",
    "Content: Numeracy": "Numeracy",
    "Content: Literacy": "Literacy",
    "Content: Play": "Play",
    "Content: Creative": "Creative",
    "Traditional models: Preschool": "Traditional models",
    "Traditional models: Child care": "Child care",
    "Management": "Management",
    "Tech": "Learning experience / Tech",
    "Workforce: Recruitment": "Recruitment",
    "Workforce: Training": "Training",
    "Workforce: Optimisation": "Optimisation",
    "Family support: General": "General support",
    "Family support: Peers": "Peers",
    "Family support: Finances": "Finances",
}

chatgpt_theme_to_keywords = {}
for key in chatgpt_theme_to_subthemes:
    chatgpt_theme_to_keywords[key] = subtheme_to_keywords_dict[
        chatgpt_theme_to_subthemes[key]
    ]

chatgpt_theme_to_keywords = {
    "Content: General": ["curriculum", "education resource"],
    "Content: Numeracy": ["numeracy", "mathematics"],
    "Content: Literacy": ["phonics", "literacy", "reading", "ebook"],
    "Content: Play": ["games", "play", "toys"],
    "Content: Creative": ["singing", "art", "painting"],
    "Traditional models: Preschool": ["pre school", "kindergarten", "montessori"],
    "Traditional models: Child care": [
        "child care",
        "nursery",
        "child minder",
        "babysitting",
    ],
    "Management": [
        "management",
        "classroom technology",
        "monitoring technology",
        "analytics",
        "waitlists",
    ],
    "Tech": [
        "robotics",
        "AI",
        "voice chat",
        "artificial intelligence",
        "machine learning",
        "simulation",
    ],
    "Workforce: Recruitment": ["recruitment", "talent aqcuisition", "hiring"],
    "Workforce: Training": ["teacher training", "skills"],
    "Workforce: Optimisation": ["retention", "wellness", "shift work"],
    "Family support: General": [
        "parents",
        "parenting advice",
        "nutrition",
        "sleep",
        "travel",
        "transport",
    ],
    "Family support: Peers": ["social network", "peer to peer"],
    "Family support: Finances": ["finances", "cash", "budgeting"],
}

for key in chatgpt_theme_to_keywords:
    print(f'"{key}" -- {", ".join(chatgpt_theme_to_keywords[key])}')

chatgpt_theme_to_keywords

processed_texts.head(1)

# add processed texts to longlist
company_texts = longlist_df[["cb_id", "relevant", "model_relevant"]].merge(
    processed_texts.rename(columns={"id": "cb_id"}), on="cb_id", how="left"
)

# +
example = "95487399-812c-d898-a435-c9494023cbbc"
# print(company_texts.query("cb_id == @example").iloc[0].description)

example = "bbe75319-e2f6-9b22-0a0e-b87b70214958"
print(company_texts.query("cb_id == @example").iloc[0].description)


# -


def num_tokens(text):
    return len(encoding.encode(text))


company_texts["num_tokens"] = company_texts.description.apply(num_tokens)

company_texts.num_tokens.sum() / 1e3 * 0.002

# +
import openai

import innovation_sweet_spots.getters.openai as getters_openai
import importlib

importlib.reload(getters_openai)
# -

openai.api_key = getters_openai.get_api_key()

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant who is labelling companies in terms of their correspondence to a predefined taxonomy. You are given a company description and asked to label it with theme. You can also choose to label it as not relevant.",
        },
        {"role": "user", "content": "Who won the world series in 2020?"},
        {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020.",
        },
        {"role": "user", "content": "Where was it played?"},
    ],
)


prompt_len = num_tokens(chat)
prompt_len

company_texts.num_tokens.sum() / 1e3 * 0.002

(company_texts.num_tokens.sum() + len(company_texts) * prompt_len) / 1e3 * 0.002

# read a txt file
from innovation_sweet_spots import PROJECT_DIR

with open(
    PROJECT_DIR / "outputs/2023_childcare/interim/openai/prompt_v0_0.txt", "r"
) as f:
    chat_0 = f.read()

chat_0

'You are a helpful assistant who is labelling companies by using predefined categories. This is for a project to map companies working on improving childcare, parental support and early years education solutions, focussed on children between 0-5 years old. Your are given keywords for each category, and the company description. You will output one or maximum two categories that best match the company description. You can also label the company as “Not relevant”. For example, we are not interested in solutions for middle or high schoolers, universities or companies not related to families or education. Here are the categories and their keywords (provided in the format “Category name” - keywords). “Content: General“ - curriculum, education resource; “Content: Numeracy“ - numeracy, mathematics, coding; “Content: Literacy“ - phonics, literacy, reading, ebook; “Content: Play“ - games, play, toys; “Content: Creative“ - singing, song, songs, art, arts, drawing, painting; “Traditional models: Preschool“ - pre school, kindergarten, montessori; “Traditional models: Child care“ - child care, nursery, child minder, babysitting; “Traditional models: Special needs” - special needs, autism, mental health; “Management“ - management, classroom technology, monitoring technology, analytics, waitlists; “Tech" - robotics, artificial intelligence, machine learning, simulation; “Workforce: Recruitment“ - recruitment, talent acquisition, hiring; “Workforce: Training“ - teacher training, skills; “Workforce: Optimisation“ - retention, wellness, shift work; “Family support: General“ - parents, parenting advice, nutrition, sleep, travel, transport; “Family support: Peers“ - social network, peer to peer; “Family support: Finances“ - finances, cash, budgeting. Here are examples of company descriptions and categories - Example 1: Description: “privacy- first speech recognition software delivers voice- enabled experiences for kids of all ages, accents, and dialects. has developed child- specific speech technology that creates highly accurate, age- appropriate and safe voice- enabled experiences for children. technology is integrated across a range of application areas including toys, gaming, robotics, as well as reading and English Language Learning . Technology is fully and GDPR compliant- offering deep learning speech recognition based online and offline embedded solutions in multiple languages. Industries: audio, digital media, events” Category: “Tech”; Example 2: Description: “is a personalized learning application to improve math skills. is a personalized learning application to improve math skills. It works by identifying a child’s level, strengths and weaknesses, and gradually progressing them at the rate that’s right for them. The application is available for download on the App Store and Google Play. Industries: accounting, finance, financial services” Category: “Content: Numeracy”. Now categorise this company: Description: “MATH 42 helps over 1.8M middle-school, high-school and college students worldwide, to understand and solve their math problems step-by-step.”'

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant who is labelling companies by using predefined categories.",
    },
    {
        "role": "system",
        "content": "'You are a helpful assistant who is labelling companies by using predefined categories. This is for a project to map companies working on improving childcare, parental support and early years education solutions, focussed on children between 0-5 years old. Your are given keywords for each category, and the company description. You will output one or maximum two categories that best match the company description. You can also label the company as “Not relevant”. For example, we are not interested in solutions for middle or high schoolers, universities or companies not related to families or education.\nHere are the categories and their keywords provided in the format Category name - keywords.\nContent: General - curriculum, education resource\nContent: Numeracy - numeracy, mathematics, coding\nContent: Literacy - phonics, literacy, reading, ebook\nContent: Play - games, play, toys\nContent: Creative - singing, song, songs, art, arts, drawing, painting\nTraditional models: Preschool - pre school, kindergarten, montessori\nTraditional models: Child care - child care, nursery, child minder, babysitting\nTraditional models: Special needs - special needs, autism, mental health\nManagement - management, classroom technology, monitoring technology, analytics, waitlists\nTech - robotics, artificial intelligence, machine learning, simulation\nWorkforce: Recruitment - recruitment, talent acquisition, hiring\nWorkforce: Training - teacher training, skills\nWorkforce: Optimisation - retention, wellness, shift work\nFamily support: General - parents, parenting advice, nutrition, feeding, sleep, travel, transport\nFamily support: Peers - social network, peer to peer\nFamily support: Finances - finances, cash, budgeting.\n\nHere are examples of company descriptions and categories - Example 1: Description: privacy- first speech recognition software delivers voice- enabled experiences for kids of all ages, accents, and dialects. has developed child- specific speech technology that creates highly accurate, age- appropriate and safe voice- enabled experiences for children. technology is integrated across a range of application areas including toys, gaming, robotics, as well as reading and English Language Learning . Technology is fully and GDPR compliant- offering deep learning speech recognition based online and offline embedded solutions in multiple languages. Industries: audio, digital media, events\nCategory: <Tech>\n\nExample 2: Description: is a personalized learning application to improve math skills. is a personalized learning application to improve math skills. It works by identifying a child’s level, strengths and weaknesses, and gradually progressing them at the rate that’s right for them. The application is available for download on the App Store and Google Play. Industries: accounting, finance, financial services.\nCategory: <Content: Numeracy>\n\nNow categorise this company: Description: MATH 42 helps over 1.8M middle-school, high-school and college students worldwide, to understand and solve their math problems step-by-step.",
    },
    {"role": "assistant", "content": "<Not relevant>"},
    {
        "role": "user",
        "content": "Description: SplashLearn is an EdTech startup company providing game-based math and reading courses to students in pre-kindergarten to grade five.",
    },
    {"role": "assistant", "content": "<Content: Numeracy> and <Content: Literacy>"},
    # {"role": "user", "content": "Description: Meludia is a web application for learning music through emotions and understanding musical compositions."},
]

texts_to_check = company_texts.astype({"model_relevant": float}).query(
    "model_relevant > 0.5"
)

df_test = texts_to_check.sample(5)

messages_complete = []
for i, row in df_test.iterrows():
    messages_complete.append(
        messages + [{"role": "user", "content": f"Description: {row['description']}"}]
    )
    # print(row['description'])
    # print(openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=messages,
    #     temperature=0.5,
    #     max_tokens=1000,
    #     ))

messages_complete

messages

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020.",
    },
    {"role": "user", "content": "Where was it played?"},
]

j = 4
print(messages_complete[j][-1])
openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages_complete[j],
    temperature=0.5,
    max_tokens=1000,
)

messages_complete
