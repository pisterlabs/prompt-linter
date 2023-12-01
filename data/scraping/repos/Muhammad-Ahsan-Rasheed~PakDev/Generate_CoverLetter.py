import cohere

co = cohere.Client('6xl3SHALT0x7HMKIhQ9A1SMCpJppaImByJJcufOq')  # This is your trial API key


def generate(role, job_description, about_company, name, about_me):
    try:
        response = co.generate(
            model='command-xlarge-nightly',
            prompt=f'Role: \nPython Developer\nJob Description:\nWe are looking for a talented Python Developer '
                   f'to join our team. As a Python Developer, you will be responsible for designing, developing, '
                   f'and testing software applications. You will work on complex projects and have the '
                   f'opportunity to work with a dynamic and talented team. The ideal candidate will have strong '
                   f'experience with Python, as well as experience with other programming '
                   f'languages.\nResponsibilities:\nDesign and develop software applications using '
                   f'Python\nCollaborate with cross-functional teams to develop new features and products\nWrite '
                   f'clean, maintainable, and efficient code\nTest and debug software applications\nEnsure '
                   f'software meets specified requirements and standards\nParticipate in code and design '
                   f'reviews\nRequirements:\nStrong experience with Python\nExperience with other programming '
                   f'languages (Java, C++, etc.)\nExcellent problem-solving skills\nKnowledge of software '
                   f'development best practices\nExcellent communication and collaboration skills\nFamiliarity '
                   f'with Agile software development methodologies\nAbility to work independently and as part of '
                   f'a team\nAbout Company:\nWe are a leading software development company, specializing in the '
                   f'development of custom software applications. Our team of experienced developers is dedicated '
                   f'to delivering innovative solutions that meet the needs of our clients. We are committed to '
                   f'providing a supportive and collaborative work environment, and to fostering the growth and '
                   f'development of our employees.\nCandidate Name: \nMuhammad Ahsan\nAbout Me:\nI am a highly '
                   f'motivated and experienced Python Developer with a passion for creating innovative software '
                   f'solutions. With a strong background in Python, I have developed a wide range of software '
                   f'applications and have experience working with a variety of programming languages. I am '
                   f'confident in my ability to work independently and as part of a team, and am always seeking '
                   f'new challenges and opportunities for growth.\nCover Letter:\nDear Hiring Manager,'
                   f'\nI am writing to express my interest in the Python Developer role at your company. As a '
                   f'passionate and experienced Python Developer, I am confident in my ability to make a valuable '
                   f'contribution to your team.\nI have extensive experience designing, developing, and testing '
                   f'software applications using Python. I have a strong background in a variety of programming '
                   f'languages, and have a proven track record of delivering high-quality software solutions. I '
                   f'am committed to using my problem-solving skills and technical expertise to deliver '
                   f'innovative and effective software solutions that meet the needs of your clients.\nI am '
                   f'impressed by the commitment to providing a supportive and collaborative work environment at '
                   f'your company, and I am eager to work with your talented team of developers. I believe that '
                   f'my passion for software development and my dedication to continuous learning and growth make '
                   f'me an excellent fit for this role.\nI would welcome the opportunity to discuss my '
                   f'qualifications further and to learn more about the opportunities available at your company. '
                   f'Thank you for considering my application.\nSincerely,\nMuhammad Ahsan\n--\nRole: \n'
                   f'{role}\nJob Description:\n{job_description}\nAbout Company:\n'
                   f'{about_company}\nCandidate Name: \n{name}\nAbout Me:\n{about_me}\nCover '
                   f'Letter:',
            max_tokens=714,
            temperature=0.6,
            k=165,
            p=0.25,
            frequency_penalty=0.03,
            presence_penalty=0,
            stop_sequences=["--"],
            return_likelihoods='GENERATION')

        return response.generations[0].text
    except Exception as e:
        return e
