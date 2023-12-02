from guidance import Program

core_program: Program = Program(
    """
{{#system~}}
You are an expert system in analyzing resumes based on their textual content.
Given the resume text, you should generate a set of attributes that represent the candidate's profile.
You need to examine the text and extract relevant information for each attribute.
You are skilled in analyzing resumes accurately and comprehensively.
{{~/system}}

{{#user~}}
You have been provided with the text of a resume for evaluation. Your task is to generate a JSON representation of the resume by populating the attributes.
Please carefully examine the text and extract the relevant information for each attribute.

Resume Text:
{{resume_text}}
do not include the following details in the resume:
- personal information (name, address, phone number, email address, etc.)
- hobbies and interests
- spoken languages
- reference links (e.g. LinkedIn, GitHub, etc.)
- any other information that is not relevant to the job application
Please generate the JSON representation of the resume.
{{~/user}}

{{#assistant~}}
{{gen 'resume_summary' temperature=0 max_tokens=2000}}
{{~/assistant}}
""",
    async_mode=True
)

extractor_function = {
    "name": "extract_resume_details",
    "description": "This function extracts the personal, education, skills, certifications, projects, experiences, and awards details from the given resume text. It does not generate any new information or text not present in the given text.",
    "parameters": {
            "type": "object",
            "properties": {
                "first_name": {
                    "type": "string",
                    "description": "The first name of the person.",
                },
                "last_name": {
                    "type": "string",
                    "description": "The last name of the person.",
                },
                "email": {
                    "type": "string",
                    "description": "The email address of the person.",
                },
                "phone": {
                    "type": "string",
                    "description": "The phone number of the person.",
                },
                "address": {
                    "type": "string",
                    "description": "The physical address of the person.",
                },
                "objective": {
                    "type": "string",
                    "description": "The career objective of the person.",
                },
                "skills": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "skill_name": {"type": "string", "description": "Name of the skill."},
                            "expertise_level": {"type": "string", "description": "Expertise level in terms of beginner, intermediate, advanced, expert."},
                        },
                        "description": "List of skills details extracted from the resume text if present else empty object.",
                    },
                    "description": "List of skills of the person.",
                },
                "experience": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Designation or title of the job."},
                            "company": {"type": "string", "description": "Company of employment."},
                            "start_date": {"type": "string", "description": "Start date of the job."},
                            "end_date": {"type": "string", "description": "End date of the job."},
                            "description": {"type": "string", "description": "Job Description."},
                        },
                        "description": "Experience details extracted from the resume text if present else empty object.",
                    },
                    "description": "List of job experiences of the person.",
                },
                "education": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "institution": {"type": "string", "description": "Institution of study."},
                            "degree": {"type": "string", "description": "Degree pursued or obtained."},
                            "field_of_study": {"type": "string", "description": "Major field of study."},
                            "start_date": {"type": "string", "description": "Start date of the study period."},
                            "end_date": {"type": "string", "description": "End date of the study period."},
                            "grade": {"type": "string", "description": "Grade, CGPA or marks obtained."},
                        },
                        "description": "Education details extracted from the resume text if present else empty object.",
                    },
                    "description": "List of educational qualifications of the person.",
                },
                "certifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Title of the certification."},
                            "issuer": {"type": "string", "description": "Issuer of the certification."},
                            "issue_date": {"type": "string", "description": "Issue date of the certification."},
                            "expiry_date": {"type": "string", "description": "Expiry date of the certification."},
                        },
                        "description": "Certification details extracted from the resume text if present else empty object.",
                    }
                },
                "projects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Title of the project."},
                            "start_date": {"type": "string", "description": "Start date of the project."},
                            "end_date": {"type": "string", "description": "End date of the project."},
                            "description": {"type": "string", "description": "Description of the project."},
                        },
                        "description": "Project details extracted from the resume text if present else empty object.",
                    },
                    "description": "List of projects done by the person if present else empty list.",
                },
                "awards": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Title of the award."},
                            "issuer": {"type": "string", "description": "Issuer of the award."},
                            "issue_date": {"type": "string", "description": "Issue date of the award."},
                        },
                        "description": "Award details extracted from the resume text if present else empty object.",
                    },
                    "description": "List of awards received by the person if present else empty list.",
                },
                "publications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Title of the publication."},
                            "publisher": {"type": "string", "description": "Publisher of the publication."},
                            "publication_date": {"type": "string", "description": "Date of the publication."},
                        },
                        "description": "Publication details extracted from the resume text if present else empty object.",
                    },
                    "description": "List of publications of the person if present else empty list.",
                },
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the reference."},
                            "contact_info": {"type": "string", "description": "Contact information of the reference."},
                            "designation": {"type": "string", "description": "Designation of the reference."},
                        },
                        "description": "Reference details extracted from the resume text if present else empty object.",
                    },
                    "description": "List of references of the person if present else empty list.",
                },
                "languages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "language": {"type": "string", "description": "Language known by the person."},
                            "fluency": {"type": "string", "description": "Fluency level in the language in terms of beginner, intermediate, advanced, expert."},
                        },
                        "description": "Language details extracted from the resume text if present else empty object.",
                    },
                    "description": "List of languages known by the person if present else empty list.",
                }
            },
        "required": [
                "first_name",
                "last_name",
                "email",
                "phone",
                "address",
                "objective",
                "skills",
                "experience",
                "education",
                "certifications",
                "projects",
                "awards",
                "publications",
                "references",
                "languages"
            ]
    }
}
