import openai


def enhance_jd(request_id, job_description):
    user_message = [
        {"role": "system", "content": """You are a HR. You are trying to create a Job Description for a openings at your company and you are not sure about the description that you have written and you need recommendation for enhancing the provided Job description maintaing the same format as it have and enhanced JD must have to be of same word length as it have it in provided JD this is mandatory. Provided JD may have many format like heading then in next line it have sub headings or paragraphs it may starts from indexing or bullet points. in enhanced JD you also have to maintain the same indexing or bullet points enhanced it as a standard Job Description.
                Input will be in the following format -
                request_id - request id willbe provided here
                Job Description - job description will be provided here

                Output should be in the form of json object with the following keys -
                request_id :  The provided request_id
                enhanced_jd : The enhanced Job Description
                """},
        {'role': 'user', 'content': f"""request_id - {request_id} Job Description-{job_description} """}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=user_message
    )
    return response['choices'][0]['message']['content']


# response in format of json object with key request id anf enhanced_jd
