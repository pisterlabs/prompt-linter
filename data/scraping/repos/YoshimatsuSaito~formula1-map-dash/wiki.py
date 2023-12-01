import ast
import os
from datetime import datetime

import openai
import wikipediaapi
from dotenv import load_dotenv

load_dotenv(".env")
openai.api_key = os.environ.get("OPENAI_API_KEY")


# Adhoc title name of wikipedia page
DICT_CIRCUIT_GPNAME = {
    "bahrain": "Bahrain",
    "jeddah": "Saudi Arabian",
    "albert_park": "Australian",
    "imola": "Emilia Romagna",
    "miami": "Miami",
    "catalunya": "Spanish",
    "monaco": "Monaco",
    "baku": "Azerbaijan",
    "villeneuve": "Canadian",
    "silverstone": "British",
    "red_bull_ring": "Austrian",
    "paulricard": "French",
    "hungaroring": "Hungarian",
    "spa": "Belgian",
    "zandvoort": "Dutch",
    "monza": "Italian",
    "marina_bay": "Singapore",
    "suzuka": "Japanese",
    "americas": "United States",
    "rodriguez": "Mexico City",
    "interlagos": "Brazilian",
    "yas_marina": "Abu Dhabi",
    "losail": "Qatar",
    "vegas": "Las Vegas",
    "shanghai": "Chinese",
}


class WikiSearcher:
    """Search wikipedia page and extract information with LLM"""

    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="My User Agent - formula1-map-dash",
        )

    def create_dict_title_past(self, gpname, years_to_create=10):
        """Create wikipedia page titles of past races"""
        last_year = datetime.now().year - 1
        list_year = list(range(last_year, last_year - years_to_create, -1))
        return {year: f"{self.create_page_title(gpname, year)}" for year in list_year}

    def create_page_title(self, gpname, year):
        """Create wikipedia page title"""
        if "Grand Prix" not in gpname:
            return f"{year} {gpname} Grand Prix"
        return f"{year} {gpname}"

    def check_page_exists(self, title):
        """Check existence of the title page"""
        title_page = self.wiki.page(title)
        if not title_page.exists():
            return False
        else:
            if title_page.title == title:
                return True
            # for redirect
            else:
                False

    def get_page_content(self, title):
        """Get wikipedia page content"""
        return self.wiki.page(title).text

    def get_condition_of_race(self, page_text, model="gpt-3.5-turbo-16k"):
        """Infer a condition of race day with LLM"""
        res = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"""
                    You are a helpful assistant to teach about a wikipedia page as shown below. 
                    
                    {page_text}
                    """
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"""
                    Understand the conditions during the final race and output as follows:
                    
                    If it can be determined that the race was conducted with no rain at all, the track surface was dry, and no red flags were raised, output as DRY.
                    In all other cases, output as RAIN.
                    Your output must be like below.
    
                    {{
                        'condition': {{your answer}}
                    }}
    
                    """
                    ),
                },
            ],
        )
        return res["choices"][0]["message"]["content"]

    def convert_to_dict(self, race_condition):
        """Convert str into dict"""
        return ast.literal_eval(race_condition)

    def get_recent_dry_race(self, gpname):
        """Get recent dry races of the grandprix"""
        if gpname not in DICT_CIRCUIT_GPNAME.values():
            raise ValueError(
                f"gpname must be one of the {DICT_CIRCUIT_GPNAME.values()}"
            )
        # Get page title to search
        dict_title = self.create_dict_title_past(gpname)
        # Get page title existed
        dict_title = {k: v for k, v in dict_title.items() if self.check_page_exists(v)}
        # Get page content
        dict_page_content = {k: self.get_page_content(v) for k, v in dict_title.items()}
        # Loop recent years
        for year, page_content in dict_page_content.items():
            # Retry {num_retry} times to get condition with LLM
            num_retry = 0
            while num_retry < 10:
                try:
                    res = self.get_condition_of_race(page_content)
                    condition = self.convert_to_dict(res)["condition"]
                    break
                except:
                    num_retry += 1
            # Proceed to next loop if all attempt was failed
            if num_retry == 10:
                continue
            # Otherwise check a condition of the year is DRY or not
            if condition == "DRY":
                return year
        # If there is no DRY race, return None
        return None
