def main():
    '''
    Generates prompts and statements of intent (SOI) for graduate school applications based on a dataset of student information. 
    The code includes two functions: skills_to_print(row) and PromptGenerator(row).

    The code also reads in an Excel file of student information. The data is then cleaned by dropping 
    several unnecessary columns, adding a leading zero to student IDs, randomly selecting a prompt version, 
    rounding age and GPA values, replacing "Unknown" and "OTHER" values with np.nan, and adding a boolean 
    columns for whether to explain certain information. These boolean columns add randomness to our prompts 
    aiming to generate prompts as diverse as possible. Later, these prompts are passed as input to the 
    OpenAI API to generate the corresponding SOI. The final output is stored into a list, and added to a Pandas DataFrame
    which is subsequently saved as a CSV file.
    '''

    import numpy as np
    import pandas as pd
    import os
    import openai

    # Setup OpenAI API
    OpenAI_API_KEY = "MyKey"
    openai.api_key = OpenAI_API_KEY
    model_id = 'gpt-3.5-turbo' 

    empty = '[EMPTY]'

    def skills_to_print(row):
        '''
        Takes in a row from the input dataset and returns a string of skills that are available for the student based on their information. 
        The function first creates a list of skills, then uses the filter() method to remove empty strings from the list of available skills, 
        and finally joins the remaining skills into a comma-separated string.
        '''
        # List of Skills
        skills_list = ['python', 'java', 'c++', 'matlab', 'sas', 'database', 'software','calculus', 'statistics', 'machine learning', 'linear algebra']
        skills_available = row[skills_list].values * skills_list
        skills_available = list(filter(lambda x: x != '', skills_available))
        return ', '.join(skills_available)

    def PromptGenerator(row):
        '''
        Takes in a row from the input dataset and generates a prompt for a graduate school application based on the student's information. 
        The function creates a number of control variables based on the input row, then generates a prompt string using string interpolation. 
        The prompt string includes information about the university, the student's intent, their race, their major, their GPA, and their skills. 
        The prompt version is also randomly selected from four variants.
        '''
        # Control Variables
        PromptVersion = row["PromptVersion"]
        InitialWord = row['FirstPromptWord']; IntentPurpose = row["IntentPurpose"]
        skillsorknowledge = row["SkillsOrKnowledge"]; SkillsList = skills_to_print(row); TalkSkills = row['TalkAboutSkills']
        race = row['race']; TalkRace = row["TalkAboutRace"]
        major = row['Major_Flag']
        gpa = row["undergrad1_gpa"]
        SOIlen = row["SOILen"]
        ProvideSOILen = row["ProvideSOILen"]

        if PromptVersion == 1:
            return f"{InitialWord} you are applying for a graduate program in Data Science at [University's Name] University. Write a statement of {IntentPurpose} that explains your reasons for pursuing this program, and how {f'your {race},' if TalkRace and not pd.isna(race) else empty}{f' your undergraduate major in {major},' if not pd.isna(major) else empty}{f' GPA of {str(gpa)},' if not pd.isna(gpa) else empty}{f' and {skillsorknowledge} ({SkillsList})' if TalkSkills and SkillsList else empty} have prepared you for success in the program.{f' The statement should have around {SOIlen} words.' if ProvideSOILen else empty}"
        elif PromptVersion == 2:
            return f"{InitialWord} you are applying for a graduate program in Data Science at [University's Name] University. Write a statement of {IntentPurpose} telling a story that explains your reasons for pursuing this program, and how {f'your {race},' if TalkRace and not pd.isna(race) else empty}{f' your undergraduate major in {major},' if not pd.isna(major) else empty}{f' GPA of {str(gpa)},' if not pd.isna(gpa) else empty}{f' and {skillsorknowledge} ({SkillsList})' if TalkSkills and SkillsList else empty} have prepared you for success in this mater's program.{f' The statement should have around {SOIlen} words.' if ProvideSOILen else empty}"
        elif PromptVersion == 3:
            return f"Write a statement of {IntentPurpose} for a master's in Data Science at [University's Name] University.{f' My undergrad is in {major},' if not pd.isna(major) else empty}{f' my GPA is {str(gpa)},' if not pd.isna(gpa) else empty}{f' and I know {SkillsList}' if TalkSkills and SkillsList else empty}.{f' The statement should have around {SOIlen} words.' if ProvideSOILen else empty}"
        elif PromptVersion == 4:
            return f"Write a statement of {IntentPurpose} for a master's in Data Science at [University's Name] University. {InitialWord}{f' you are an undergrad in {major},' if not pd.isna(major) else empty}{f' your GPA is {str(gpa)},' if not pd.isna(gpa) else empty}{f' and you are skilled in {SkillsList}' if TalkSkills and SkillsList else empty}.{f' The statement should have around {SOIlen} words.' if ProvideSOILen else empty}"
        else:
            return np.nan

    # List all columns to use
    columns_to_use = ["ID\n(delete)", "age_at_submission", "undergrad1_gpa", 'python', 'java', 'c++', 'matlab', 'sas', 'database', 
                        'software','calculus', 'statistics', 'machine learning', 'linear algebra', "race", "Major_Flag"]
    
    # Read students' information to further prompt generation
    students_info = pd.read_excel("merged-data-without-TOEFL-final.xlsx")[columns_to_use]
    # Feature engineering and feature generation to add randomness and sense to the prompts
    students_info["PromptVersion"] = np.random.choice([1,2,3,4], size=len(students_info))
    students_info['undergrad1_gpa'] = students_info['undergrad1_gpa'].map(lambda x: round(x,2) if x > 3.40 else np.nan)
    students_info.replace({'Unknown':np.nan},inplace=True)
    students_info.replace({'OTHER':np.nan},inplace=True)
    students_info["TalkAboutAge"] = np.random.choice([True, False], size=len(students_info),p=[.05,.95])
    students_info["TalkAboutRace"] = np.random.choice([True, False], size=len(students_info),p=[.2,.8])
    students_info["TalkAboutSkills"] = np.random.choice([True, False], size=len(students_info),p=[.95,.05])
    students_info["SkillsOrKnowledge"] = np.random.choice(['skills', 'knowledge'], size=len(students_info))
    students_info["FirstPromptWord"] = np.random.choice(["Imagine", "Asumme", "Think","Let's say"], size=len(students_info))
    students_info["IntentPurpose"] = np.random.choice(["intent", "purpose"], size=len(students_info))
    students_info["SOILen"] = np.random.randint(300, 700, size=len(students_info))
    students_info["ProvideSOILen"] = np.random.choice([True, False], size=len(students_info),p=[.2,.8])

    # Call Prompt generator function
    students_info["Prompt"] = students_info.apply(PromptGenerator,axis=1)
    # Format prompts the resulted prompts to generate SOI
    students_info["Prompt"] = students_info["Prompt"].map(lambda x: x.replace(empty, "")).replace("  ", " ")
    
    # Use OpenAI API to generate SOI
    GeneratedSOIs = []
    ## Loop through dataframe and create files
    for _, row in students_info.iterrows():
        Prompt = row["Prompt"]
        conversation = []
        # Input prompt to chatbot (GPT3.5)
        conversation.append({"role": "system", "content": Prompt})
        # Generate SOI
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=conversation)
        
        GeneratedSOI = response.choices[-1].message.content.lstrip().replace("\n\n","\n")
        GeneratedSOIs.append(GeneratedSOI)
  
    students_info["AIGenerated"] = GeneratedSOIs
    students_info[["ID\n(delete)", "Prompt", "AIGenerated"]].to_csv(os.path.join(os.getcwd(), "PresentationGeneratedSOIs.csv"), index=False)

    print(students_info)
    
if __name__ == "__main__":
    main()
