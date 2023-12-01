import guidance
import os

if os.environ["OPENAI_API_KEY"] is None:
    print("OPENAI_API_KEY environment variable not set")
    os.exit(1)

guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo-0301", api_key=os.environ["OPENAI_API_KEY"])

def extract_info(user_input):
    
    # ? Example input
    # user_input = """Hello, my name is John Silver and I am 25 years old. I live in Palo Alto right now. I graduated from UCLA in 2010 with a Bachelor's Degree in Computer Science, while in UCLA I developed some Java aplications for the school. I have started my professional  careerin October of 2011. I have worked as a software engineer at Google. I worked on numerous projects including Speech to Text, Text to Speech,Large Language Models, and many more. We developed a special algorithm that can generate a song from a given picture. After working atGoogle for 5 years, I switched to Apple as a Team Lead. I have been working at Apple for 3 years now. I'm currently working on a productthat lets users generate a CV from their LinkedIn profile. I have a personal websitehttps://www.johnsilver.com  and my LinkedIn profile is https://www.linkedin.com/in/johnsilver. My email is me@johnsilver.com. I've used Python, C++,  and JavaScript. Delved into the MySQLand NoSQL. Used a bit React and Flutter. I have a dog named Rex. I love to play tennis and I'm a big fan of the Lakers."""
    
    # ? Example output
    # user_info = {'first_name': 'John', 'last_name': 'Silver', 'email': 'me@johnsilver.com', 'phone': '', 'address': 'Palo Alto', 'personal_website': 'https://www.johnsilver.com', 'linkedin_link': 'https://www.linkedin.com/in/johnsilver', 'github_link': '', 'previous_education_info': [{'school_name': 'UCLA', 'degree': "Bachelor's Degree in Computer Science", 'start_date': '09/06', 'end_date': '06/10', 'description': 'Developed some Java aplications for the school.'}], 'previous_workplace_info': [{'company_name': 'Apple', 'position': 'Team Lead', 'start_date': '10/16', 'end_date': '', 'description': 'Working on a product that lets users generate a CV from their LinkedIn profile.'}, {'company_name': 'Google', 'position': 'Software Engineer', 'start_date': '10/11', 'end_date': '10/16', 'description': 'Worked on numerous projects including Speech to Text, Text to Speech, Large Language Models, and many more. Developed a special algorithm that can generate a song from a given picture.'}], 'previously_used_programming_languages': ['Python', 'C++', 'JavaScript'], 'previously_used_frameworks': ['React', 'Flutter'], 'previously_used_databases': ['MySQL', 'NoSQL'], 'human_languages': []}

    
    generate_one_json = guidance("""
    {{#system}}
    You extract and parse important information from given plain text to JSON format. 
    Place "" if the information is not provided. Current year is 2023, convert relative dates to MM/YY format, like; 11/19. 
    Sort work and education experience by start date in reverse chronological order. Paraphrase the description if not professional enough.
    Schema for the JSON output;
    {
        "first_name": "",
        "last_name": "",
        "email": "",
        "phone": "",
        "address": "",
        "personal_website": "",
        "linkedin_link": "",
        "github_link": "",
        "previous_education_info": [
            {
                "school_name": "",
                "degree": "",
                "start_date": "",
                "end_date": "",
                "description": ""
            }]
        "previous_workplace_info": [
            {
                "company_name": "",
                "position": "",
                "start_date": "",
                "end_date": "",
                "description": ""
            }]
        "previously_used_programming_languages": [],
        "previously_used_frameworks": [],
        "previously_used_databases": [],
        "human_languages": [],
    }
    {{/system}}
    
    {{#user}}
    Text to parse:
    "{{user_info}}"
    {{/user}}
    
    {{#assistant}}
    {{gen 'extracted_information' stop="\n\n"}}
    {{/assistant}}
    """)

    cv_information = generate_one_json(
        user_info=user_input
    )

    json_data = eval(cv_information["extracted_information"])

    return json_data


def suggest_improvements(user_info_json):
    possible_improvements = guidance("""
    {{#system}}
    You improve the given CV information by returning which information is missing and what can be improved. Find ambiguities 
    and suggest the user to add more details or delete unnecessary information. If the workplace or education entries' descriptions are
    not detailed enough or missing, suggest the user to add more details and give example. Bullet points are preferred. If the number
    of programming languages and frameworks are less than 3, suggest the user to add more. Group personal information suggestions into one.
    Print only JSON. Return suggestions in JSON format;
    {
        suggestions: [
            {
                "importancy": enum("none", "low", "medium", "high"),
                "type": "workplace",
                "description": "Add more details to your work experience at Google. For example, you can add bullet points like; ..."
            }
    }
    {{/system}}
    
    {{#user}}
    User information:
    "{{user_info}}"
    {{/user}}
    
    {{#assistant}}
    {{gen 'suggestions' stop="\n\n"}}
    {{/assistant}}
    """)
    
    improvements = possible_improvements(
        user_info=str(user_info_json)
    )

    json_data = eval(improvements["suggestions"])
    
    # sort based on importancy
    json_data = sorted(json_data["suggestions"], key=lambda k: k['importancy'])

    return json_data

def improve_cv(user_info_json, suggestions):
    make_improvements = guidance("""
    {{#system}} 
    You improve the given CV information and suggestions by the user. Change necessary information and add missing information.
    Print improved CV information in JSON format only.
    {{/system}}
    
    {{#user}}
    User information:
    "{{user_info}}"
    
    Wanted improvements:
    "{{improvements}}"
    
    {{/user}}
    
    {{#assistant}}
    {{gen 'suggestions' stop="\n\n"}}
    {{/assistant}}
    """)
    
    improved_cv = make_improvements(
        user_info=str(user_info_json),
        improvements=suggestions
    )

    json_data = eval(improved_cv["suggestions"])

    return json_data

def generate_cover_letter(user_info, job_listing):
    cover_later_template = guidance("""
    {{#system}} 
    You generate a cover letter for the given job listing and user information. 
    Return relevant information with cover letter in JSON format only.
    Include why you want to use for that company, try to include previous work experience and education.
    Template for output JSON:
    {
        "first_name": "",
        "last_name": "",
        "email": "",
        "phone": "",
        "address": "",
        "recipient": "",
        "opening": "",
        "closing": "",
        "letter_text": ""
    }
    {{/system}}
    
    {{#user}}
    User information:
    "{{user_info}}"
    
    Job listing:
    "{{job_listing}}"
    
    {{/user}}
    
    {{#assistant}}
    {{gen 'cover_letter' stop="\n\n"}}
    {{/assistant}}
    """)
    
    cover_letter_info = cover_later_template(
        user_info=str(user_info),
        job_listing=job_listing
    )

    json_data = eval(cover_letter_info["cover_letter"])

    return json_data
