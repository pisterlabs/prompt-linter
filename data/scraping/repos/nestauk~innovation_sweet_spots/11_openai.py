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
import pandas as pd

pd.set_option("display.max_colwidth", 1000)

import innovation_sweet_spots.getters.google_sheets as gs
from innovation_sweet_spots.getters.preprocessed import (
    get_preprocessed_crunchbase_descriptions,
)
import utils


# +
import innovation_sweet_spots.utils.openai as openai_utils
import openai

openai.api_key = openai_utils.get_api_key()

import importlib

importlib.reload(openai_utils)

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

# add processed texts to longlist
company_texts = (
    longlist_df[["cb_id", "relevant", "model_relevant"]]
    .merge(processed_texts.rename(columns={"id": "cb_id"}), on="cb_id", how="left")
    .assign(num_tokens=lambda df: df.description.apply(openai_utils.num_tokens))
)

company_texts.num_tokens.sum() / 1e3 * 0.002

(company_texts.num_tokens.sum() + len(company_texts) * prompt_len) / 1e3 * 0.002

# read a txt file
from innovation_sweet_spots import PROJECT_DIR

with open(
    PROJECT_DIR / "outputs/2023_childcare/interim/openai/prompt_v0_0.txt", "r"
) as f:
    chat_0 = f.read()

prompt = [
    {
        "role": "system",
        "content": "You are a helpful assistant who is labelling companies by using predefined categories.",
    },
    {
        "role": "user",
        "content": "'You are a helpful assistant who is labelling companies by using predefined categories. This is for a project to map companies working on improving childcare, parental support and early years education solutions, focussed on children between 0-5 years old. Your are given keywords for each category, and the company description. You will output one or maximum two categories that best match the company description. You can also label the company as “Not relevant”. For example, we are not interested in solutions for middle or high schoolers; universities; healthcare; or companies not related to families or education.\n\nHere are the categories and their keywords provided in the format Category name - keywords.\nContent: General - curriculum, education content, resource\nContent: Numeracy - numeracy, mathematics, coding\nContent: Literacy - phonics, literacy, reading, ebook\nContent: Play - games, play, toys\nContent: Creative - singing, song, songs, art, arts, drawing, painting\nTraditional models: Preschool - pre school, kindergarten, montessori\nTraditional models: Child care - child care, nursery, child minder, babysitting\nTraditional models: Special needs - special needs, autism, mental health\nManagement - management, classroom technology, monitoring technology, analytics, waitlists\nTech - robotics, artificial intelligence, machine learning, simulation\nWorkforce: Recruitment - recruitment, talent acquisition, hiring\nWorkforce: Training - teacher training, skills\nWorkforce: Optimisation - retention, wellness, shift work\nFamily support: General - parents, parenting advice, nutrition, feeding, sleep, travel, transport\nFamily support: Peers - social network, peer to peer\nFamily support: Finances - finances, cash, budgeting.\n\nHere are examples of company descriptions and categories.\n\nExample 1: Description: privacy- first speech recognition software delivers voice- enabled experiences for kids of all ages, accents, and dialects. has developed child- specific speech technology that creates highly accurate, age- appropriate and safe voice- enabled experiences for children. technology is integrated across a range of application areas including toys, gaming, robotics, as well as reading and English Language Learning . Technology is fully and GDPR compliant- offering deep learning speech recognition based online and offline embedded solutions in multiple languages. Industries: audio, digital media, events\nCategory: <Tech>\n\nExample 2: Description: is a personalized learning application to improve math skills. is a personalized learning application to improve math skills. It works by identifying a child’s level, strengths and weaknesses, and gradually progressing them at the rate that’s right for them. The application is available for download on the App Store and Google Play. Industries: accounting, finance, financial services.\nCategory: <Content: Numeracy>\n\nNow categorise this company: Description: The company helps over 1.8M middle-school, high-school and college students worldwide, to understand and solve their math problems step-by-step.",
    },
    {"role": "assistant", "content": "<Not relevant>"},
    {
        "role": "user",
        "content": "Description: The company  is an EdTech startup company providing game-based math and reading courses to students in pre-kindergarten to grade five.",
    },
    {"role": "assistant", "content": "<Content: Numeracy> and <Content: Literacy>"},
    {
        "role": "user",
        "content": "Description: The company is a global digital- first entertainment company for kids. The company is a global entertainment company that creates and distributes inspiring and engaging stories to expand kids’ worlds and minds. Founded in 2018, with offices in and, The company creates, produces and publishes thousands of minutes of video and audio content every month with the goal of teaching compassion, empathy and resilience to kids around the world.",
    },
    {"role": "assistant", "content": "<Content: General>"},
    # {"role": "user", "content": "Description: Meludia is a web application for learning music through emotions and understanding musical compositions."},
]

for p in prompt:
    print(f"{p['role']}:")
    print(p["content"])
    print("----")

texts_to_check = pd.concat(
    [
        # take all rows with model_relevant > 0.5
        company_texts.astype({"model_relevant": float}).query("model_relevant > 0.5"),
        # take a sample of 500 rows with model_relevant < 0.5
        company_texts.astype({"model_relevant": float})
        .query("model_relevant < 0.5")
        .sample(500, random_state=42),
    ],
    ignore_index=True,
)


cb_ids = texts_to_check.cb_id.to_list()

from innovation_sweet_spots import PROJECT_DIR

pd.DataFrame(data={"cb_id": cb_ids}).to_csv(
    PROJECT_DIR / "outputs/2023_childcare/interim/openai/cb_ids_v2023_03_14.csv",
    index=False,
)

cb_ids_to_check = pd.read_csv(
    PROJECT_DIR / "outputs/2023_childcare/interim/openai/cb_ids_v2023_03_14.csv"
).cb_id.to_list()

company_texts.query("cb_id in @cb_ids_to_check")

texts_to_check

texts_to_check

df_test = texts_to_check.sample(5)
df_test

queries = []
for i, row in df_test.iterrows():
    queries.append(
        prompt + [{"role": "user", "content": f"Description: {row['description']}"}]
    )

chatgpt_outputs = []

j = 3
print(queries[j][-1])
chatgpt_output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=queries[j],
    temperature=0.5,
    max_tokens=1000,
)

(chatgpt_output.to_dict().update({"cb_id": 111}))

# convert chatgpt_output dict into str
chatgpt_output_str = chatgpt_output.to_dict()
str(chatgpt_output_str)

chatgpt_output_str

# +
import csv

fields = [
    "id",
    "object",
    "created",
    "model",
    "usage",
    "choices",
    "cb_id",
]

with open("test.csv", "a") as f:
    chatgpt_output_str.update({"cb_id": df_test.cb_id.iloc[j]})
    writer = csv.DictWriter(f, fields)
    # writer.writeheader()
    writer.writerow(chatgpt_output_str)
# -

queries[j][-1]["content"]

longlist_df
