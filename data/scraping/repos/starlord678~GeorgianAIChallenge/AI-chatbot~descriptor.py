"""import requests

openaiKey = "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"

while True:
    finalNums = []
    userInput = input("Enter a name for analysis: ").strip()
    name = userInput

    weights = [0.4, 0.3, 0.3]  # Adjust weights based on importance of each aspect

    for i in range(3):
        if i == 0:
            aspect = "startup pitches from podcasts, YouTube, and TED Talks"
        elif i == 1:
            aspect = "LinkedIn Profiles, Biographies, and Case Studies, Forbes"
        elif i == 2:
            aspect = "Engagement Metrics from Social Media"
        elif i == 2:
            aspect = "Engagement Metrics from Social Media"

        userInput = f"DON'T REPLY WITH TEXT IN YOUR REPLY. YOUR ANSWER SHOULD BE A NUMBER BETWEEN 1 AND 100: Evaluate {name}'s {aspect}, by comparing to other visionary leaders. Use internet whatever available information provided about this person. if you don't have access to something, omit it"

        message = [{'role': 'user', 'content': userInput}]
        headers = {'Authorization': f'Bearer {openaiKey}'}
        data = {'model': 'gpt-3.5-turbo', 'temperature': 0.3, 'messages': message}
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

        finalNums.append({
            'aspect': aspect,
            'rating': int(response.json()['choices'][0]['message']['content']),
            'weight': weights[i]
        })

    # Calculate weighted average
    total_weighted_sum = sum(num['rating'] * num['weight'] for num in finalNums)
    total_weight = sum(num['weight'] for num in finalNums)
    weighted_average = total_weighted_sum / total_weight

    print(f"Weighted Average Rating for {name}: {weighted_average}")

    exit_condition = input("Do you want to analyze another name? (yes/no): ").lower()
    if exit_condition != 'no':
        break
        """
"""

import requests

openaiKey = "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"

def get_ratings(name):
    prompts = [
        "Generate a higher number the more likely Sam Altman is to become a successful startup entrepreneur, based on his speeches and presentations, relative to other visionary leaders.",
        "Generate a higher number the more likely Sam Altman is to become a successful startup entrepreneur, based on his LinkedIn Profiles, Biographies, and Forbes Fortune 500, relative to other visionary leaders.",
        "Generate a higher number the more likely Sam Altman is to become a successful startup entrepreneur, based on his social media metrics and engagement online, relative to other visionary leaders.",
        "Generate a higher number the more likely Sam Altman is to become a successful startup entrepreneur, based on his Company Insights from Crunchbase and AngelList, relative to other visionary leaders."
    ]

    # Adjust weights for each aspect
    weights = [
        {'aspect': 'speeches_and_presentations', 'weight': 0.3},
        {'aspect': 'linkedin_and_biographies', 'weight': 0.2},
        {'aspect': 'social_media_metrics', 'weight': 0.25},
        {'aspect': 'company_insights', 'weight': 0.25}
    ]

    ratings = []
    for i, prompt in enumerate(prompts):
        userInput = f"INT1,INT2,INT3,INT4 SHOULD BE BETWEEN 1 AND 100  AND YOUR ANSWER SHOULD BE IN THE FOLLOWING FORMAT: '[INT1,INT2,INT3,INT4], DESCRIPTION': {prompt} {name}."
        message = [{'role': 'user', 'content': userInput}]
        headers = {'Authorization': f'Bearer {openaiKey}'}
        data = {'model': 'gpt-3.5-turbo', 'temperature': 0.3, 'messages': message}
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

        # Extract the rating from the user's response
        rating_str = response.json()['choices'][0]['message']['content'].split('[', 1)[1].split(']')[0]
        rating = [max(1, min(100, int(num.strip()))) for num in rating_str.split(',')]

        ratings.append({
            'aspect': weights[i]['aspect'],
            'ratings': rating,
            'weight': weights[i]['weight']
        })

    return ratings

while True:
    name = input("Enter a name for analysis: ").strip()

    ratings = get_ratings(name)

    # Calculate weighted sum for each aspect
    weighted_sums = []
    for i in range(4):
        total_weighted_sum = sum(num * ratings[i]['weight'] for num in ratings[i]['ratings'])
        weighted_sums.append(total_weighted_sum)

    # Calculate the average of the weighted sums
    average_weighted_sum = sum(weighted_sums) / len(weighted_sums)

    # Display the average of the weighted sums
    print(f"Average Weighted Sums for {name}: {average_weighted_sum}")

    exit_condition = input("Do you want to analyze another name? (yes/no): ").lower()
    if exit_condition != 'yes':
        break
"""
"""
import openai

# Set your OpenAI GPT-3 API key
openai.api_key = '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'

def generate_leader_profile(leader_name):
    # Template for the prompt
    prompt = f"In each of the fields, fill in information about {leader_name}\n\n"\
             "Professional Background:\n\n"\
             "Education:\n"\
             "Work History:\n\n"\
             "Key Achievements and Milestones:\n\n"\
             "Notable Accomplishments:\n"\
             "Industry Recognition:\n\n"\
             "Leadership Style and Philosophy:\n\n"\
             "Leadership Approach:\n"\
             "Philosophy and Vision:\n\n"\
             "Innovation and Contributions:\n\n"\
             "Technological Contributions:\n"\
             "Impact on Industry Trends:\n\n"\
             "Public Persona and Communication Skills:\n\n"\
             "Communication Style:\n"\
             "Public Speaking Engagements:\n\n"\
             "Advisory and Mentorship Roles:\n\n"\
             "Advisory Positions:\n"\
             "Mentorship Activities:\n\n"\
             "Company Culture and Employee Satisfaction:\n\n"\
             "Company Culture:\n"\
             "Employee Satisfaction:\n\n"\
             "Global Impact and Philanthropy:\n\n"\
             "Philanthropic Initiatives:\n"\
             "Global Impact:\n\n"\
             "Media Perception and Public Image:\n\n"\
             "Media Coverage:\n"\
             "Public Sentiment:\n\n"\
             "Current and Future Ventures:\n\n"\
             "Current Roles:\n"\
             "Future Ventures:\n\n"\
             "Challenges Faced and Resilience:\n\n"\
             "Challenges Overcome:\n"\
             "Resilience:"

    # Make the API call
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can use other engines as well
        prompt=prompt,
        max_tokens=1500,  # Adjust as needed
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated text from the API response
    generated_text = response['choices'][0]['text']

    return generated_text

# Example usage
leader_name = "Bill Gates"
profile = generate_leader_profile(leader_name)
print(profile)
"""
import requests

openaiKey = "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"
responses = []  # List to store responses

def get_ratings(name):
    prompt_2 = f"In each of the fields, fill in information about {name}\n\n"\
             "Professional Background:\n\n"\
             "Education:\n"\
             "Work History:\n\n"\
             "Key Achievements and Milestones:\n\n"\
             "Notable Accomplishments:\n"\
             "Industry Recognition:\n\n"\
             "Leadership Style and Philosophy:\n\n"\
             "Leadership Approach:\n"\
             "Philosophy and Vision:\n\n"\
             "Innovation and Contributions:\n\n"\
             "Technological Contributions:\n"\
             "Impact on Industry Trends:\n\n"\
             "Public Persona and Communication Skills:\n\n"\
             "Communication Style:\n"\
             "Public Speaking Engagements:\n\n"\
             "Advisory and Mentorship Roles:\n\n"\
             "Advisory Positions:\n"\
             "Mentorship Activities:\n\n"\
             "Company Culture and Employee Satisfaction:\n\n"\
             "Company Culture:\n"\
             "Employee Satisfaction:\n\n"\
             "Global Impact and Philanthropy:\n\n"\
             "Philanthropic Initiatives:\n"\
             "Global Impact:\n\n"\
             "Media Perception and Public Image:\n\n"\
             "Media Coverage:\n"\
             "Public Sentiment:\n\n"\
             "Current and Future Ventures:\n\n"\
             "Current Roles:\n"\
             "Future Ventures:\n\n"\
             "Challenges Faced and Resilience:\n\n"\
             "Challenges Overcome:\n"\
             "Resilience:"

    prompts = [
        "Generate a higher number the more likely the entrepreneur is to become a successful startup entrepreneur, based on his speeches and presentations, relative to other visionary leaders.",
        "Generate a higher number the more likely the entrepreneur is to become a successful startup entrepreneur, based on his LinkedIn Profiles, Biographies, and Forbes Fortune 500, relative to other visionary leaders.",
        "Generate a higher number the more likely the entrepreneur is to become a successful startup entrepreneur, based on his social media metrics and engagement online, relative to other visionary leaders.",
        "Generate a higher number the more likely the entrepreneur is to become a successful startup entrepreneur, based on his Company Insights from Crunchbase and AngelList, relative to other visionary leaders."
    ]

    weights = [
        {'aspect': 'speeches_and_presentations', 'weight': 0.3},
        {'aspect': 'linkedin_and_biographies', 'weight': 0.2},
        {'aspect': 'social_media_metrics', 'weight': 0.25},
        {'aspect': 'company_insights', 'weight': 0.25}
    ]

    ratings = []
    for i, prompt in enumerate(prompts):
        user_input = f"INT1,INT2,INT3,INT4 SHOULD BE BETWEEN 1 AND 100.  for DESCRIPTION, fill in the fields with the entrepreneur's background. ANSWER SHOULD BE IN THE FOLLOWING FORMAT: '[INT1,INT2,INT3,INT4], DESCRIPTION': {prompt} {name}."
        message = [{'role': 'user', 'content': user_input}]
        headers = {'Authorization': f'Bearer {openaiKey}'}
        data = {'model': 'gpt-3.5-turbo', 'temperature': 0.3, 'messages': message}
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        responses.append(response.json())  # Save the response

        rating_str = response.json()['choices'][0]['message']['content'].split('[', 1)[1].split(']')[0]
        rating = [max(1, min(100, int(num.strip()))) for num in rating_str.split(',')]

        ratings.append({
            'aspect': weights[i]['aspect'],
            'ratings': rating,
            'weight': weights[i]['weight']
        })

    return ratings

while True:
    name = input("Enter a name for analysis: ").strip()

    #print("\nGenerated Leader Profile:")
    ratings = get_ratings(name)

    weighted_sums = []
    for i in range(4):
        total_weighted_sum = sum(num * ratings[i]['weight'] for num in ratings[i]['ratings'])
        weighted_sums.append(total_weighted_sum)

    average_weighted_sum = sum(weighted_sums) / len(weighted_sums)

    print(f"\nAverage Weighted Sums for {name}: {average_weighted_sum}")

    exit_condition = input("\nDo you want to analyze another name? (yes/no): ").lower()
    if exit_condition != 'yes':
        break

# Now you can access the responses list outside the loop if needed
print(responses)


