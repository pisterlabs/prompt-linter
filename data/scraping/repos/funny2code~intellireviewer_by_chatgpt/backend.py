import re
import openai 

openai.api_key = os.getenv("OPENAI_API_KEY")

def call_openai(prompt, max_tokens=2300):
    messages = [{"role": "system", "content": "you are career advisor"}]
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.get("choices")[0]['message']['content']

def industry_trends(text, details=''):

    prompt = """Analyze the employee's performance review to identify growth-oriented topics for 1:1 meetings with their manager. Focus on areas benefiting from guidance, feedback, and support. Present prioritized discussion topics in a clear and organized format:
            Topic 1 (related to growth objectives)
            Topic 2 (related to growth objectives)
            Topic 3 (related to growth objectives)
            Topic 4 (related to growth objectives)
            Topic 5 (related to growth objectives)
            Please format it nicely with line breaks to make it more readable
            """
    if details:
        prompt += f"use this details to grab more ideas about the user in oreder to recommend courses {details}"

    response = call_openai(f"{text}\n\n{prompt}")

    return response

def SMART(text):
    prompt = """Analyze the employee's performance review to set SMART goals for the next year. Provide an action plan with the following elements:
            Milestones: Break goals into manageable milestones.
            Resources: Identify relevant resources for skill development.
            Time management: Offer strategies for effective time allocation.
            Accountability: Suggest methods for progress tracking.
            Potential obstacles: Anticipate challenges and provide strategies to overcome them.
            Support network: Encourage building a network of colleagues, mentors, and peers.
            A structured action plan helps the employee achieve their goals and advance in their career.
            Please format it nicely with line breaks to make it more readable
            """

    response = call_openai(f"{text}\n\n{prompt}")
    return_response = ''
    temp_list = response.split('•')
    if len(temp_list) > 2:
        for item in temp_list:
            if item != '':
                return_response += f"* {item}"
        return return_response


    return response 

def soft_skills(text, details=''):

    prompt = """Based on the employee's performance review, provide personalized tips for soft skill development and topics for professional growth. List these recommendations in a prioritized order:
            Tip 1 (related to strengths/growth areas)
            Tip 2 (related to strengths/growth areas)
            Tip 3 (related to strengths/growth areas)
            Please format it nicely with line breaks to make it more readable
            """

    response = call_openai(f"{text}\n\n{prompt}")
    if details:
        prompt += f"use this details to grab more ideas about the user in oreder to recommend courses {details}"
    return_response = ''
    temp_list = response.split('•')
    if len(temp_list) > 2:
        for item in temp_list:
            if item != '':
                return_response += f"* {item}"
        return return_response

    return response 

def get_details(text):
    prompt = """As an attentive reader, meticulously extract the employee's name, organization, manager, and evaluator from the performance review document.
        Write the employee's name, organization, manager, and evaluator, formatted as a Markdown list.
        """
    
    response = call_openai(f"{text}\n\n{prompt}")
    response = re.sub(r"[0-9].|•", "\n-", response)
    slice = response.find("-")
    res = response[slice:]
    print("Details: ", res)
    return res

def career_dev_ops(text, details=''):
    prompt = """Analyze the employee's performance review and identify career development opportunities.
        Write the possible opportunities, formatted as a Markdown list.
        """
    if details:
        prompt += f"use this details to grab more ideas about the user in oreder to recommend courses {details}"
    response = call_openai(f"{text} \n {prompt}")
    response = re.sub(r"[0-9].|•", "\n-", response)
    slice = response.find("-")
    res = response[slice:]
    print("Career: ", res)
    return res

def recommend_courses(text, details =""):
    prompt = """Analyze the employee's performance review to identify strengths and growth areas. 
        Write 3 relevant courses from Udemy or LinkedIn Learning with its url to help them maximize potential, formatted as a Markdown list.
        """
    
    if details:
        prompt += f"use this details to grab more ideas about the user in oreder to recommend courses {details}"
    response = call_openai(f"{text}\n\n{prompt}")
    response = re.sub(r"[0-9].|•", "\n-", response)
    slice = response.find("-")
    res = response[slice:]
    print("Courses: ", res)
    return res

def highlight_action_items(text, details =''):
    prompt = """Analyze the employee's performance review to identify areas where they can contribute to the team's goals. 
        Write the targeted measures to create a high-performance environment and recommend prioritized action items for the next year based on their performance review, formatted as a Markdown list.
        """

    if details:
        prompt += f"use this details to grab more ideas about the user in oreder to recommend courses {details}"
    response = call_openai(f"{text}\n\n{prompt}")
    response = re.sub(r"[0-9].|•", "\n-", response)
    slice = response.find("-")
    res = response[slice:]
    print("Action Items: ", res)
    return res