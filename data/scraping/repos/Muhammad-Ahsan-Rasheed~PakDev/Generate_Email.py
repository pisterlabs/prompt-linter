import cohere

co = cohere.Client('6xl3SHALT0x7HMKIhQ9A1SMCpJppaImByJJcufOq')  # This is your trial API key


def generate_email(from_name, to_name, subject, context):
    try:
        response = co.generate(
            model='command-xlarge-nightly',
            prompt=f'From: John\nTo: Doe (HR)\nSubject: Application for job as a Data Analyst.\nContext: I am interested in '
                   f'applying for a job opening at a company. I want to introduce myself, highlight my relevant experience '
                   f'and qualifications, and request for interview for further discussion. \nBody: Dear Doe,\nI hope this '
                   f'email finds you well. My name is John, and I am writing to express my interest in the Data Analyst '
                   f'role at Tesco. I came across this opportunity on https://www.tesco-careers.com/ and was immediately '
                   f'impressed by the company\'s commitment to make customers lifestyle simpler.\nI am a highly motivated '
                   f'and results-driven individual with 66 years of experience in IT and Financial relevant field. In my '
                   f'current role as a Data Analyst, I have honed my skills in Data Cleaning, Exploring, Analysis and '
                   f'Visualizing and I am confident that my experience and qualifications make me a strong fit for this '
                   f'role.\nI have attached my resume for your review, and I would be grateful for the opportunity to '
                   f'discuss my application further in an interview. Thank you for considering my application.\nI look '
                   f'forward to hearing back from you.\n\nRegards,\nJohn\n--\nFrom: Jane\nTo:  David ('
                   f'Administrator)\nSubject: Application for Admission in The University of Arizona.\nContext: I am '
                   f'applying for a master\'s degree at The University of Arizona via email to the administrator, '
                   f'including academic background, achievements, and experience with transcripts and test scores.\nBody: '
                   f'Dear David,\nI hope this email finds you in good health. My name is Jane, and I am writing to express '
                   f'my interest in pursuing a master’s degree at The University of Arizona. I have been following the '
                   f'university for a long time and I am impressed by the quality of education and the reputation it holds '
                   f'in the industry.\nI have completed my graduation in computer science from Quest University with a GPA '
                   f'of 3.8/4.0. In addition to my academic achievements, I have also been actively involved in leading IT '
                   f'societies in university and have gained valuable skills and experience in leading.\nI am very excited '
                   f'about the opportunity to further my education at the University of Arizona, and I am confident that I '
                   f'will be able to make the most of the resources and opportunities available to me. I have attached my '
                   f'transcripts and test scores for your review, and I would be grateful for the opportunity to provide '
                   f'any additional information that may be required.\nThank you for considering my application. I look '
                   f'forward to the opportunity to contribute to the academic community at the University of '
                   f'Arizona.\nSincerely, Jane\n--\nFrom: Sohay\nTo: James\nSubject: Application for the Recruiter Position '
                   f'at LinkedIn.\nContext: I am expressing an interest in the Recruiter role at LinkedIn, with 4 years of '
                   f'experience in recruitment, skills in communication and interpersonal relations, and commitment to the '
                   f'company. Also attached resume and cover letter for review.\nBody: Dear James,\nI hope you are well and '
                   f'having read this email. My name is Sohay, and I am writing to express my interest in the Recruiter '
                   f'position at LinkedIn. I have been following the company for some time and I am impressed by its '
                   f'commitment to provide online job facilities and multi-level personal developments. \nI have 4 years of '
                   f'experience in the recruitment field and have a proven track record of successfully sourcing and hiring '
                   f'top talent for a variety of industries. My expertise in communication skills, big picture thinking, '
                   f'readability, confidence, and patience, has allowed me to build strong relationships with both '
                   f'candidates and hiring managers, and I am confident that I can bring my skills and experience to the '
                   f'team at LinkedIn.\nI have attached my resume and a cover letter for your review. I would be grateful '
                   f'for the opportunity to discuss my application further in an interview. Thank you for considering my '
                   f'application.\nI look forward to hearing back from you.\nRegards,\nSohay\n--\nFrom: Jessica\nTo: Pablo '
                   f'Vilem (Program Manager)\nSubject: Application for the Microsoft Ambassadorship Program.\nContext: I am '
                   f'applying for the Microsoft Ambassadorship Program and have sent an email to Pablo Vilem. Highlights of '
                   f'passion for IT, and impressive social media following with beliefs that my passion and eager to '
                   f'represent the company and share love for its products.\nBody: Dear Pablo Vilem,\nI hope this email '
                   f'finds you doing well. My name is Jessica, and I am writing to express my interest in the Microsoft '
                   f'Ambassadorship Program. I have been a fan of this company for a long time, and I am impressed by the '
                   f'quality of its products and the passion of its community.\nI am a highly motivated and enthusiastic '
                   f'individual with a strong online presence and a passion for Information Technology. I have 20k+ '
                   f'followers on LinkedIn and 1M subscribers on YouTube and I regularly engage with my followers by '
                   f'sharing Volunteer and Leadership skills content.\nI believe that my passion for ambassadorship and my '
                   f'strong online presence would make me a valuable asset to the Microsoft Ambassadorship Program. I would '
                   f'be honored to represent this company and to share my passion for its products with my followers and '
                   f'the wider community. \nThank you for considering my application. I look forward to the opportunity to '
                   f'contribute to the success of this Ambassadorship Program.\nThanks & regards,\nPablo Vilem\n--\nFrom: '
                   f'{from_name}\nTo: {to_name}\nSubject: {subject}\nContext: {context}\nBody:',
            max_tokens=700,
            temperature=0.9,
            k=175,
            p=0.25,
            frequency_penalty=0.03,
            presence_penalty=0,
            stop_sequences=["--"],
            return_likelihoods='GENERATION')
    except Exception as e:
        return e

    return response.generations[0].text
