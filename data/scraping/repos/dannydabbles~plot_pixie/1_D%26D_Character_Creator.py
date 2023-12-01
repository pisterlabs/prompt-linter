import os
import sys
import json
import boto3
import random
import openai
import streamlit as st
import pandas as pd
import requests
from uuid import uuid4
from fpdf import FPDF
from PIL import Image
from langchain.utilities.dalle_image_generator import DallEAPIWrapper

# Set configuration variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION_NAME = os.environ.get('AWS_REGION_NAME') 
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
CLOUDFRONT_URL = os.environ.get('CLOUDFRONT_URL')

# Constants
CURRENT_DIRECTORY = os.getcwd()
IMAGE_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "images")
CHARACTER_SHEET_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "character_sheets")
DATA_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "data")
PAGES_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "pages")
MAX_TOKENS = 1500
DEBUG = False

# Ensure directories exist
os.makedirs(IMAGE_DIRECTORY, exist_ok=True)
os.makedirs(CHARACTER_SHEET_DIRECTORY, exist_ok=True)
os.makedirs(DATA_DIRECTORY, exist_ok=True)

# List options
CLASS_LIST = [
    "Barbarian",
    "Bard",
    "Cleric",
    "Druid",
    "Fighter",
    "Monk",
    "Paladin",
    "Ranger",
    "Rogue",
    "Sorcerer",
    "Warlock",
    "Wizard"
]
RACE_LIST = [
    "Dragonborn",
    "Dwarf",
    "Elf",
    "Gnome",
    "Half-Elf",
    "Half-Orc",
    "Halfling",
    "Human",
    "Tiefling",
    "Aarakocra",
    "Aasimar",
    "Bugbear",
    "Firbolg",
    "Goblin",
    "Goliath",
    "Hobgoblin",
    "Kenku",
    "Kobold",
    "Lizardfolk",
    "Orc",
    "Tabaxi",
    "Triton",
    "Yuan-ti Pureblood",
    "Genasi",
    "Changeling",
    "Kalashtar",
    "Shifter",
    "Warforged",
    "Centaur",
    "Loxodon",
    "Minotaur",
    "Simic Hybrid",
    "Vedalken",
    "Verdan",
    "Leonin",
    "Satyr"
]

# Set up for script
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION_NAME
)
# Set the page configuration at the very top of the script
st.set_page_config(page_title="D&D Character Creator", page_icon="🐉")
# Sidebar
st.sidebar.title("D&D Character Creator 🐉")
st.sidebar.write("Your AI D&D Character Creator.")

st.write("# D&D Character Creator! 🐉")
st.write("###### *Note: Character generation can take 60+ seconds.  Please be patient, true magic can't be rushed.*")

def character_name_to_id(character_name: str) -> str:
    """
    Convert a character name to a unique ID.

    Args:
        character_name (str): The name of the character.

    Returns:
        str: The unique ID of the character.
    """

    return character_name.replace(" ", "_").lower() + "_" + str(uuid4())

def upload_file_to_s3(file_path: str, s3_key: str) -> str:
    """
    Uploads a file to S3.
    
    Args:
        file_path (str): The local path of the file to upload.
        s3_key (str): The desired S3 key (path) for the uploaded file.
        
    Returns:
        str: The URL of the file in S3.
    """
    s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
    return f"{CLOUDFRONT_URL}/{s3_key}"

def get_character_age() -> int:
    """
    Generate a weighted random age for a character between 0 and 500 years old.

    Returns:
        int: The age of the character.
    """
    age_weights = [0.07, 0.1, 0.43, 0.2, 0.1, 0.05, 0.05]
    age_choices = [
        random.randint(1, 13), random.randint(13, 17), random.randint(18, 35),
        random.randint(36, 50), random.randint(50, 120), random.randint(121, 1000),
        random.randint(1001, sys.maxsize)
    ]
    age = random.choices(age_choices, weights=age_weights)[0]
    return age

def get_character_examples() -> list:
    """
    Load character_examples.json file in current directory.

    Returns:
        list: List of character examples.
    """
    character_examples = []
    with open(os.path.join(PAGES_DIRECTORY, "character_examples.json")) as f:
        character_examples = json.loads(f.read())
    return character_examples

def get_character_data(character: dict) -> str:
    """
    Query the ChatGPT API to fill out missing character data based on provided data.
    
    Args:
        character (dict): Dictionary containing character attributes.

    Returns:
        str: Generated character description.
    """
    examples = get_character_examples()

    if DEBUG:
        print(f"Examples: {json.dumps(examples)}")

    messages = [
    {"role": "system", "content": "You are a dedicated assistant specializing in D&D character creation."},
    {"role": "system", "content": "Users will provide incomplete character sheets in JSON format. Complete them by filling in missing details, ensuring no keys have empty values except sometimes N/A. Characters should be both unique and playable within D&D 5e rules."},
    {"role": "system", "content": "Ensure all generated stats strictly adhere to D&D 5e rules for the specified level, class, and race of the character. Mark any fields that don't make sense for the given character's level, race, or class with N/A."},
    {"role": "system", "content": "Provide a 'portrait_prompt' for each character. This prompt should be a detailed, evocative description tailored for DALL-E to visualize the character."},
    {"role": "system", "content": f"Use the following character sheet template as a guide:\n\n{json.dumps(examples[0])}"},
    {"role": "user", "content": "Fill out D&D 5e character sheets for me, ensuring stats are accurate for the character's level, class, and race--or N/A if not appropriate. Do not alter the age, level, class, or race fields if non-empty. Populate or mark spell fields as N/A based on the character's class and level. Skip any spells that aren't appropriate for the given character's level, class, or race according to D&D 5e rules. Make sure the JSON is valid."},
    {"role": "assistant", "content": f"{json.dumps(examples[1])}"},
    {"role": "user", "content": "Great! Now, craft a brand new character sheet for me."},
    {"role": "assistant", "content": f"{json.dumps(examples[2])}"}
    ]

    # Loop through each character example as a user/assistant message pair
    for example in examples[3:]:
        messages.append({"role": "user", "content": "Craft another unique character sheet for me, ensuring stats align with D&D 5e rules and paying special attention to the portrait prompt."})
        messages.append({"role": "assistant", "content": f"{json.dumps(example)}"})

    messages.append({"role": "user", "content": f"Complete this character sheet for me, ensuring it's a valid JSON and stats are consistent with D&D 5e rules:\n\n{json.dumps(character)}"})

    if DEBUG:
        print(f"Messages: {json.dumps(messages)}")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k", messages=messages, max_tokens=MAX_TOKENS
    )
    if DEBUG:
        print(f"Response: {json.dumps(response)}")
    result = json.loads(response.choices[0].message.content)
    if DEBUG:
        print(f"Result: {json.dumps(result)}")
    return result

def save_dalle_image_to_s3(image_url: str, character_id: str, portrait_num: int) -> str:
    """
    Saves the DALL·E generated image to the S3.

    Args:
        image_url (str): URL of the image generated by DALL·E.
        character_id (str): ID of the character to be used in filename.
        portrait_num (int): The number of the portrait for this character.

    Returns:
        str: The URL for the uploaded file.
    """
    response = requests.get(image_url)
    response.raise_for_status()

    # Use character name in filename
    s3_key = f"sheets/{character_id}/portrait_{portrait_num}.png"
    tmp_filename = "/tmp/temp_image_{uuid4()}"

    with open(tmp_filename, "wb") as file:
        file.write(response.content)
    
    # Upload to S3
    cloudfront_url = upload_file_to_s3(tmp_filename, s3_key)

    # Remove file
    os.remove(tmp_filename)

    return cloudfront_url

def generate_portrait(prompt: str, character_id: str, portrait_num: int) -> str:
    """
    Generate a portrait based on the prompt using DALL-E and save it locally.
    
    Args:
        prompt (str): The prompt for DALL-E to generate an image.
        character_id (str): The ID of the character for filename generation.
        portrait_num (int): The number of the portrait for this character.

    Returns:
        str: Filepath where the portrait is saved.
    """
    image_url = DallEAPIWrapper(n=portrait_num, size="256x256").run(prompt)
    
    return save_dalle_image_to_s3(image_url, character_id, portrait_num)

def create_pdf_character_sheet(character_id: str, character: dict, portrait_filenames: list) -> str:
    """
    Create a PDF character sheet based on the provided character data.

    Args:
        character_id (str): The ID of the character for filename generation.
        character (dict): The character data to be used in the character sheet.
        portrait_filenames (list): List of filenames for the character portraits.

    Returns:
        str: The URL for the uploaded file.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Set colors for headers and fills
    header_color = (100, 100, 100)  # Dark gray
    fill_color = (220, 220, 220)  # Light gray
    
    # Define fonts
    header_font = ("Arial", 'B', 18)
    sub_header_font = ("Arial", 'B', 14)
    label_font = ("Arial", 'B', 12)
    value_font = ("Arial", '', 12)
    
    # Function to add a section header
    def add_section_header(title, y_offset=None):
        if y_offset:
            pdf.ln(y_offset)
        pdf.set_fill_color(*header_color)
        pdf.set_font(*header_font)
        pdf.cell(0, 10, txt=title, ln=True, fill=True, align='L')
    
    # Function to add a sub-section header
    def add_sub_section_header(title):
        pdf.set_fill_color(*fill_color)
        pdf.set_font(*sub_header_font)
        pdf.cell(0, 10, txt=title, ln=True, fill=True, align='L')
    
    # Function to add key-value pairs with multi_cell for text wrapping
    def add_key_value(key, value, w1=45, w2=45, ln=False):
        pdf.set_font(*label_font)
        pdf.cell(w1, 8, txt=f"{key.replace('_', ' ').capitalize()}:", align='R')
        pdf.set_font(*value_font)
        current_y = pdf.get_y()
        current_x = pdf.get_x()
        pdf.multi_cell(w2, 8, txt=f"{value}")
        if not ln:
            pdf.set_xy(current_x + w2, current_y)
    
    # Character Name as Header
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 20, txt=character['name'], ln=True, align='C')

    # Portraits
    for filename in portrait_filenames:
        try:
            # Show the image centered
            pdf.image(filename, x=60, y=None, w=90)
            pdf.ln(5)
        except Exception as e:
            print(f"Error adding image {filename} to PDF: {e}")
    
    # Separate description to handle spacing and wrapping properly
    description = character["description"]
    description_height = len(description) / 72 + 1
    add_key_value("description", character["description"], w1=50, w2=140, ln=True) 
    
    # Basic Info
    add_section_header("Basic Info")
    basic_info_keys = ['level', 'pronouns', 'orientation', 'race', 'class', 'alignment', 'background', 'age', 'height', 'weight', 'eyes', 'skin', 'hair', 'experience_points']
    for idx, key in enumerate(basic_info_keys):
        add_key_value(key, character[key], ln=(idx % 2 != 0))
    
    # Character Stats
    add_section_header("Character Stats", y_offset=10)
    stats_keys = ['armor_class', 'hit_points', 'speed', 'strength', 'dexterity', 'constitution', 'intelligence', 'wisdom', 'charisma', 'passive_wisdom_perception', 'inspiration', 'proficiency_bonus']
    for idx, key in enumerate(stats_keys):
        add_key_value(key, character[key], ln=(idx % 2 != 0))
    
    # Saving Throws & Skills
    add_section_header("Saving Throws", y_offset=10)
    throws_keys = ['strength_save', 'dexterity_save', 'constitution_save', 'intelligence_save', 'wisdom_save', 'charisma_save']
    for idx, key in enumerate(throws_keys):
        add_key_value(key, character[key], ln=(idx % 2 != 0))
    
    # Skills
    add_section_header("Skills", y_offset=10)
    skills_list = ["Acrobatics", "Animal Handling", "Arcana", "Athletics", "Deception", "History", "Insight", "Intimidation", "Investigation", "Medicine", "Nature", "Perception", "Performance", "Persuasion", "Religion", "Sleight of Hand", "Stealth", "Survival"]
    for idx, skill in enumerate(skills_list):
        skill_key = f"skills_{skill.lower().replace(' ', '_')}"
        add_key_value(skill, character[skill_key], ln=(idx % 2 != 0))
    
    # Proficiencies & Languages
    add_section_header("Proficiencies & Languages", y_offset=10)
    add_key_value("Languages", character['languages'], w1=50, w2=140, ln=True)
    add_key_value("Proficiencies", character['proficiencies'], w1=50, w2=140, ln=True)
    
    # Character Traits
    add_section_header("Character Traits", y_offset=10)
    trait_keys = ['personality_traits', 'ideals', 'bonds', 'flaws', 'character_appearance', 'character_backstory']
    for key in trait_keys:
        add_key_value(key, character[key], w1=50, w2=140, ln=True)
    
    # Features & Traits
    add_section_header("Features & Traits", y_offset=10)
    add_key_value("Features & Traits", character['features_traits'], w1=50, w2=140, ln=True)
    
    # Equipment & Treasure
    add_section_header("Equipment & Treasure", y_offset=10)
    add_key_value("Equipment", character['equipment'], w1=50, w2=140, ln=True)
    add_key_value("Treasure", character['treasure'], w1=50, w2=140, ln=True)
    
    # Attacks & Spellcasting
    add_section_header("Attacks & Spellcasting", y_offset=10)
    add_key_value("Details", character['attacks_spellcasting'], w1=50, w2=140, ln=True)
    add_key_value("Spellcasting Ability", character['spellcasting_ability'], w1=50, w2=140, ln=True)
    add_key_value("Spell Save DC", character['spell_save_dc'], w1=50, w2=140, ln=True)
    add_key_value("Spell Attack Bonus", character['spell_attack_bonus'], w1=50, w2=140, ln=True)

    # Spells Known
    add_section_header("Spells Known", y_offset=10)
    for level in range(1, 10):
        spells_text = character[f'{level}_level_spells']
        if not spells_text:
            spells_text = "N/A"
        add_key_value(f"{level} Level Spells", character[f'{level}_level_spells'], w1=50, w2=140, ln=True)
    
    pdf_file_path = os.path.join(CHARACTER_SHEET_DIRECTORY, f"{character['name'].replace(' ', '_')}_{uuid4()}.pdf")
    pdf.output(pdf_file_path)

    # Upload PDF to S3 and return the CloudFront URL
    s3_key = f"sheets/{character_id}/{character['name']}.pdf"
    s3_client.upload_file(pdf_file_path, S3_BUCKET_NAME, s3_key)
    pdf_file_url = f"{CLOUDFRONT_URL}/{s3_key}"

    # Clean up the PDF file
    os.remove(pdf_file_path)

    return pdf_file_url

def default_character() -> dict:
    """
    Returns a default character object with all fields set to empty strings or empty lists.

    Returns:
        dict: A default character object.
    """
    # Grab an example character and clear out the values
    character_examples = get_character_examples()

    default_character = character_examples[0]
    for key in default_character:
        if isinstance(default_character[key], list):
            default_character[key] = []
        else:
            default_character[key] = ""

    return default_character

def build_form(character: dict) -> st.form:
    """
    Constructs the interactive form in the Streamlit application to gather and display 
    character details for a D&D character.

    Args:
        character (dict): A dictionary containing the character's details. The dictionary keys 
                          correspond to various character attributes, and this function updates
                          the dictionary in-place based on user input.

    Returns:
        st.form: The Streamlit form object containing all the character input fields.
    """
    # If there's a valid PDF path in the session state, display the download button
    if 'pdf_url' in st.session_state and st.session_state.pdf_url:
        st.markdown(f"Download your [character sheet PDF here]({st.session_state.pdf_url})")

    form = st.form(key='character_form', clear_on_submit=True)

    with form:
        if 'portrait_filenames' in st.session_state and st.session_state.portrait_filenames:
            for filename in st.session_state.portrait_filenames:
                st.image(filename, caption=f"Portrait of {character['name']}", use_column_width=True)

        # Level, Name, Description
        character['level'] = st.text_input("Level (Recommended)", character.get('level', ''))
        character['name'] = st.text_input("Character Name (Recommended)", character['name'])
        character['description'] = st.text_area("Description (Recommended)", character['description'])

        # Basic Info
        with st.expander("Basic Info (Optional)"):
            cols = st.columns(2)
            character['pronouns'] = cols[0].text_input("Pronouns", character.get('pronouns', ''), key='basic_pronouns_input')
            character['orientation'] = cols[0].text_input("Orientation", character.get('orientation', ''), key='basic_orientation_input')
            character['race'] = cols[0].text_input("Race", character['race'], key='basic_race_input')
            character['class'] = cols[0].text_input("Class", character['class'], key='basic_class_input')
            character['alignment'] = cols[0].text_input("Alignment", character['alignment'], key='basic_alignment_input')
            character['background'] = cols[0].text_input("Background", character['background'], key='basic_background_input')
            character['age'] = cols[1].text_input("Age", character['age'], key='basic_age_input')
            character['height'] = cols[1].text_input("Height", character.get('height', ''), key='basic_height_input')
            character['weight'] = cols[1].text_input("Weight", character.get('weight', ''), key='basic_weight_input')
            character['eyes'] = cols[1].text_input("Eyes", character.get('eyes', ''), key='basic_eyes_input')
            character['skin'] = cols[1].text_input("Skin", character.get('skin', ''), key='basic_skin_input')
            character['hair'] = cols[1].text_input("Hair", character.get('hair', ''), key='basic_hair_input')
            character['experience_points'] = st.text_input("Experience Points", character.get('experience_points', ''), key='basic_experience_points_input')

        # Stats
        with st.expander("Stats (Optional)"):
            cols = st.columns(6)
            character['armor_class'] = cols[0].text_input("Armor Class", character.get('armor_class', ''), key='stats_armor_class_input')
            character['hit_points'] = cols[1].text_input("Hit Points", character.get('hit_points', ''), key='stats_hit_points_input')
            character['speed'] = cols[2].text_input("Speed", character.get('speed', ''), key='stats_speed_input')
            character['strength'] = cols[3].text_input("Strength", character.get('strength', ''), key='stats_strength_input')
            character['dexterity'] = cols[4].text_input("Dexterity", character.get('dexterity', ''), key='stats_dexterity_input')
            character['constitution'] = cols[5].text_input("Constitution", character.get('constitution', ''), key='stats_constitution_input')
            character['intelligence'] = cols[0].text_input("Intelligence", character.get('intelligence', ''), key='stats_intelligence_input')
            character['wisdom'] = cols[1].text_input("Wisdom", character.get('wisdom', ''), key='stats_wisdom_input')
            character['charisma'] = cols[2].text_input("Charisma", character.get('charisma', ''), key='stats_charisma_input')
            character['passive_wisdom_perception'] = cols[3].text_input("Passive Wisdom (Perception)", character.get('passive_wisdom_perception', ''), key='stats_passive_wisdom_input')
            character['inspiration'] = cols[4].text_input("Inspiration", character.get('inspiration', ''), key='stats_inspiration_input')
            character['proficiency_bonus'] = cols[5].text_input("Proficiency Bonus", character.get('proficiency_bonus', ''), key='stats_proficiency_bonus_input')

        # Saving Throws
        with st.expander("Saving Throws (Optional)"):
            cols = st.columns(6)
            character['strength_save'] = cols[0].text_input("Strength", character.get('strength_save', ''), key='saves_strength_input')
            character['dexterity_save'] = cols[1].text_input("Dexterity", character.get('dexterity_save', ''), key='saves_dexterity_input')
            character['constitution_save'] = cols[2].text_input("Constitution", character.get('constitution_save', ''), key='saves_constitution_input')
            character['intelligence_save'] = cols[3].text_input("Intelligence", character.get('intelligence_save', ''), key='saves_intelligence_input')
            character['wisdom_save'] = cols[4].text_input("Wisdom", character.get('wisdom_save', ''), key='saves_wisdom_input')
            character['charisma_save'] = cols[5].text_input("Charisma", character.get('charisma_save', ''), key='saves_charisma_input')

        # Skills
        with st.expander("Skills (Optional)"):
            cols = st.columns(4)
            skills_list = ["Acrobatics", "Animal Handling", "Arcana", "Athletics", "Deception", "History", "Insight", "Intimidation", "Investigation", "Medicine", "Nature", "Perception", "Performance", "Persuasion", "Religion", "Sleight of Hand", "Stealth", "Survival"]
            for idx, skill in enumerate(skills_list):
                character[f"skills_{skill.lower().replace(' ', '_')}"] = cols[idx % 4].text_input(skill, character.get(f"skills_{skill.lower().replace(' ', '_')}", ''), key=f"skills_{skill.replace(' ', '_').lower()}_input")

        # Character Traits
        with st.expander("Character Traits (Optional)"):
            character['personality_traits'] = st.text_area("Personality Traits", character.get('personality_traits', ''), key='traits_personality_input')
            character['ideals'] = st.text_area("Ideals", character.get('ideals', ''), key='traits_ideals_input')
            character['bonds'] = st.text_area("Bonds", character.get('bonds', ''), key='traits_bonds_input')
            character['flaws'] = st.text_area("Flaws", character.get('flaws', ''), key='traits_flaws_input')

        # Attacks and Spellcasting
        with st.expander("Attacks and Spellcasting (Optional)"):
            character['attacks_spellcasting'] = st.text_area("Details", character.get('attacks_spellcasting', ''), key='attacks_details_input')

        # Other Proficiencies and Languages
        with st.expander("Proficiencies and Languages (Optional)"):
            character['languages'] = st.text_input("Languages", character.get('languages', ''))
            character['proficiencies'] = st.text_input("Proficiencies", character.get('proficiencies', ''))

        # Equipment
        with st.expander("Equipment (Optional)"):
            character['equipment'] = st.text_area("Details", character.get('equipment', ''), key='equipment_details_input')

        # Features and Traits
        with st.expander("Features and Traits (Optional)"):
            character['features_traits'] = st.text_area("Details", character.get('features_traits', ''), key='features_details_input')

        # Allies and Organizations
        with st.expander("Allies and Organizations (Optional)"):
            character['allies_organizations'] = st.text_area("Details", character.get('allies_organizations', ''), key='allies_organizations_details_input')

        # Spellcasting
        with st.expander("Spellcasting (Optional)"):
            cols = st.columns(3)
            character['spellcasting_ability'] = cols[0].text_input("Spellcasting Ability", character.get('spellcasting_ability', ''), key='spells_ability_input')
            character['spell_save_dc'] = cols[1].text_input("Spell Save DC", character.get('spell_save_dc', ''), key='spells_save_dc_input')
            character['spell_attack_bonus'] = cols[2].text_input("Spell Attack Bonus", character.get('spell_attack_bonus', ''), key='spells_attack_bonus_input')

        # Spells Known
        with st.expander("Spells Known (Optional)"):
            for level in range(1, 10):
                character[f'{level}_level_spells'] = st.text_area(f"Level {level} Spells", character.get(f'{level}_level_spells', ''), key=f'{level}_level_spells_input')

        # Character Appearance
        with st.expander("Character Appearance (Optional)"):
            character['character_appearance'] = st.text_area("Details", character.get('character_appearance', ''), key='character_appearance_details_input')

        # Character Backstory
        with st.expander("Character Backstory (Optional)"):
            character['character_backstory'] = st.text_area("Details", character.get('character_backstory', ''), key='backstory_details_input')

        # Treasure
        with st.expander("Treasure (Optional)"):
            character['treasure'] = st.text_area("Details", character.get('treasure', ''), key='treasure_details_input')

    return form

def save_character_json_to_s3(character_id: str, character: dict, unprocessed: bool = False) -> None:
    """
    Saves the character JSON to S3.

    Args:
        character_id (str): The character ID.
        character (dict): The character data to save.
        unprocessed (bool, optional): Whether or not the character is unprocessed. Defaults to False.

    Returns:
        None
    """
    # Save the character data to a JSON file
    with st.spinner('Saving character data...'):
        try:
            s3_key = None
            if unprocessed:
                s3_key = f"sheets/{character_id}/_unprocessed.json"
            else:
                s3_key = f"sheets/{character_id}/{character['name']}.json"
            character_data = character.copy()
            character_data['id'] = character_id
            character_json = json.dumps(character_data)
            s3_client.put_object(Body=character_json, Bucket=S3_BUCKET_NAME, Key=s3_key)
        except Exception as e:
            st.error(f"Error saving character data: {str(e)}")

def calculate_modifier(score: int) -> int:
    """
    Calculate the ability modifier based on the ability score.

    Args:
        score (int): The ability score.

    Returns:
        int: The ability modifier.
    """
    return (score - 10) // 2

def validate_and_fix_character_sheet(character: dict) -> tuple:
    """
    Validate and fix the character sheet.

    Args:
        character (dict): The character data.

    Returns:
        Tuple[bool, str, dict]: A tuple containing the validation status, error message, and character data.
    """
    # Validate level
    level = None
    try:
        level = int(character['level'])
        if level < 1:
            character['level'] = "1"
    except ValueError:
        return False, f"Character level is not a valid integer: {character['level']}", character

    # Check if any X_level_spell keys have an empty value
    for key in character.keys():
        if key.endswith("_level_spells"):
            if character[key] == "":
                character[key] = "N/A"

    # Check if spell stats are empty
    if character['spell_save_dc'] == "":
        character['spell_save_dc'] = "N/A"
    if character['spellcasting_ability'] == "":
        character['spellcasting_ability'] = "N/A"
    if character['spell_attack_bonus'] == "":
        character['spell_attack_bonus'] = "N/A"

    # Get the character's level and class
    experience_points = int(str(character["experience_points"]).replace(",", ""))

    # Verify experience points
    if level == 1:
        if 0 >= experience_points or experience_points > 300:
            character["experience_points"] = 0
    elif level == 2:
        if 300 >= experience_points or experience_points > 900:
            character["experience_points"] = 300
    elif level == 3:
        if 900 >= experience_points or experience_points > 2700:
            character["experience_points"] = 900
    elif level == 4:
        if 2700 >= experience_points or experience_points > 6500:
            character["experience_points"] = 2700
    elif level == 5:
        if 6500 >= experience_points or experience_points > 14000:
            character["experience_points"] = 6500
    elif level == 6:
        if 14000 >= experience_points or experience_points > 23000:
            character["experience_points"] = 14000
    elif level == 7:
        if 23000 >= experience_points or experience_points > 34000:
            character["experience_points"] = 23000
    elif level == 8:
        if 34000 >= experience_points or experience_points > 48000:
            character["experience_points"] = 34000
    elif level == 9:
        if 48000 >= experience_points or experience_points > 64000:
            character["experience_points"] = 48000
    elif level == 10:
        if 64000 >= experience_points or experience_points > 85000:
            character["experience_points"] = 64000
    elif level == 11:
        if 85000 >= experience_points or experience_points > 100000:
           character["experience_points"] = 85000
    elif level == 12:
        if 100000 >= experience_points or experience_points > 120000:
            character["experience_points"] = 100000
    elif level == 13:
        if 120000 >= experience_points or experience_points > 140000:
            character["experience_points"] = 120000
    elif level == 14:
        if 140000 >= experience_points or experience_points > 165000:
            character["experience_points"] = 140000
    elif level == 15:
        if 165000 >= experience_points or experience_points > 195000:
            character["experience_points"] = 165000
    elif level == 16:
        if 195000 >= experience_points or experience_points > 225000:
            character["experience_points"] = 195000
    elif level == 17:
        if 225000 >= experience_points or experience_points > 265000:
            character["experience_points"] = 225000
    elif level == 18:
        if 265000 >= experience_points or experience_points > 305000:
            character["experience_points"] = 265000
    elif level == 19:
        if 305000 >= experience_points or experience_points > 355000:
            character["experience_points"] = 305000
    elif level >= 20:
        if 355000 >= experience_points:
            character["experience_points"] = 355000

    # Iterate through spell levels
    for spell_level in range(1, 10):  # Levels 1 through 9
        spell_key = f"{spell_level}_level_spells"

        # Check if the spell level exists in the character sheet
        if spell_key in character:
            # Get the spells of that level
            spells = character[spell_key]

            # Check if the character level allows spells of this level
            if level < spell_level:
                # Clear all spells above the character's level
                character[spell_key] = "N/A"

    if level < 17:
        spell_key = "9_level_spells"
        if character[spell_key] != "N/A":
            character[spell_key] = "N/A"

    # List of essential keys
    essential_keys = ["name", "level", "class", "strength", "dexterity", "constitution", 
                      "intelligence", "wisdom", "charisma", "proficiency_bonus", 
                      "armor_class", "hit_points", "speed"]
    
    # 1. Check for missing essential keys
    for key in essential_keys:
        if key not in character:
            return False, f"Missing essential key: {key}", character
   
    # 2. Ensure core stats are between 1 and 30
    for key in ['strength', 'dexterity', 'constitution', 'intelligence', 'wisdom', 'charisma']:
        if not 1 <= int(character[key]) <= 30:
            return False, f"Invalid value for {key}: {character[key]}", character
    
    # 3. Ensure proficiency bonus is consistent with the character's level
    proficiency = int(character['proficiency_bonus'].replace("+", ""))
    if level <= 4:
        expected_proficiency = 2
    elif 5 <= level <= 8:
        expected_proficiency = 3
    elif 9 <= level <= 12:
        expected_proficiency = 4
    elif 13 <= level <= 16:
        expected_proficiency = 5
    else:  # 17 <= level <= 20
        expected_proficiency = 6
    if proficiency != expected_proficiency:
        character['proficiency_bonus'] = f"+{expected_proficiency}"

    # 4. Check Armor Class (basic check)
    if not 10 <= int(character['armor_class']) <= 30:
        return False, f"Invalid Armor Class: {character['armor_class']}", character

    # Return false if any fields have empty values
    for key in character.keys():
        if character[key] == "":
            return False, f"Empty value for {key}", character
    
    # If all checks pass
    return True, "Character sheet is valid!", character

def main():
    """
    Main function for the D&D Character Creator app.
    """

    with st.expander("How to Use the D&D Character Creator: A Poem"):
        st.markdown("""
        *Fill what you like,*  
        *or leave a space,*  
        *my magic will craft a matching face.*  
        **For novices and masters, heed this rhyme,**  
        **Follow these steps, one at a time:**

        1. Begin with your character's name so neat,  
           A personal touch makes the journey complete.
        2. Choose a class, be it rogue or mage,  
           Each has its skills, so set your stage.
        3. Adjust the fields, pick traits that fit,  
           Strength, wisdom, or charm? You commit.
        4. Add equipment, potions, and gear,  
           Essentials for quests, far and near.


        *Remember, options can be left in a haze,*  
        *And the system's magic will craft in its own ways.*  
        *Review your creation, see it come alive,*  
        *Edit, save, and share, watch your character thrive.*  

        *So dive into this app, and let stories unfurl,*  
        *Craft, play, and explore, in this new open world.*  
        """)

    if 'character' not in st.session_state or not st.session_state.character:
        st.session_state.character = default_character()

    character = st.session_state.character.copy()

    form = build_form(character)

    st.write("###### *Note: Character generation can take 60+ seconds.  Please be patient, true magic can't be rushed.*")

    if form.form_submit_button("Generate Character Sheet", use_container_width=True):
        portrait_placeholder = st.empty()
        save_button_placeholder = st.empty()

        with st.spinner('Generating character data...'):
            main_keys = ["age", "level", "class", "race"]
            
            all_fields_empty = all(value == "" for value in character.values())
            all_main_fields_empty = all(character[key] == "" for key in main_keys)
            some_main_fields_empty = any(character[key] == "" for key in main_keys)
            some_non_main_fields_empty = any(character[key] != "" for key in character.keys() if key not in main_keys)
            all_non_main_fields_empty = all(character[key] == "" for key in character.keys() if key not in main_keys)

            # If any of the main keys are empty, set some random defaults
            if all_fields_empty or (some_main_fields_empty and all_non_main_fields_empty):
                for key in main_keys:
                    if character[key] == "":
                        if key == "age":
                            character[key] = get_character_age()
                        elif key == "level":
                            character[key] = random.randint(1, 20)
                        elif key == "class":
                            character[key] = random.choice(CLASS_LIST)
                        elif key == "race":
                            character[key] = random.choice(RACE_LIST)

            # Generate the character if any data is missing
            some_values_missing = any(value == "" for value in character.values())
            character_sheet_changed = any(character[key] != st.session_state.character[key] for key in character.keys())
            if some_values_missing or character_sheet_changed:
                # Retry once if something breaks
                rabbit_hole_depth = 2
                try:
                    for idx in range(rabbit_hole_depth):
                        # Get character data from API, then munge and validate it
                        character = get_character_data(character)
                        valid, error_message, character = validate_and_fix_character_sheet(character)
                        if valid:
                            break
                        # If we've already tried to fix the character sheet
                        if idx == rabbit_hole_depth - 1:
                            st.error(f"Error validating character sheet: {error_message}")
                            return
                except Exception as e:
                    st.error(f"Error generating character data: {str(e)}")
                    return

        # Generate the character ID
        character_id = character_name_to_id(character['name'])
        save_character_json_to_s3(character_id, st.session_state.character, unprocessed=True)

        with st.spinner('Generating PDF character sheet...'):
            try:
                st.session_state.portrait_filenames = []

                # Generate portrait prompts and portrait
                num_portraits = 1
                # TODO: Add slider for number of portraits maybe?
                # num_portraits = st.slider("Number of Portraits", 1, 5)
                for portrait_num in range(num_portraits):
                    portrait_prompt = character.get("portrait_prompt", "")

                    # Check if portrait_prompt is empty
                    if not portrait_prompt:
                        st.write("No portrait prompt provided. Skipping portrait generation.")
                        continue

                    try:
                        with st.spinner('Generating character portrait...'):
                            st.session_state.portrait_filenames.append(generate_portrait(portrait_prompt, character_id, num_portraits))
                    except Exception as e:
                        st.error(f"Error generating portrait: {str(e)}")

                # Create PDF character sheet and save
                pdf_url = create_pdf_character_sheet(character_id, character, st.session_state.portrait_filenames).replace(" ", "%20")

                # Save the path to the PDF in the session state
                st.session_state.pdf_url = pdf_url

            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                return

        # Save the character data to a JSON file
        save_character_json_to_s3(character_id, character)

        st.session_state.character = character
        st.experimental_rerun()


if __name__ == "__main__":
    main()
